import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        # Small note: unfortunate name clash with torch.nn
        # nn here refers to the specific architecture file found in
        # codebase/models/nns/*.py
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # 待办：在这里修改/完善代码
        # 计算负的证据下界（ELBO）及其 KL 分解和重构（Rec）分解
        #
        # 注意 nelbo = kl + rec
        #
        # 输出结果应该都是标量
        ################################################################################

        # 使用 encoder 计算潜在变量 z 的均值和方差
        latent_mean, latent_variance = self.enc.encode(x)
    
        # 使用先验分布的均值和方差，计算 KL 散度
        prior_mean = self.z_prior[0].expand(latent_mean.shape)
        prior_variance = self.z_prior[1].expand(latent_variance.shape)
        kl_divergences = ut.kl_normal(latent_mean, latent_variance, prior_mean, prior_variance)
        kl_divergence = torch.mean(kl_divergences)

        # 通过重参数化技巧，从潜在变量 z 的分布中抽样
        sampled_latent = ut.sample_gaussian(latent_mean, latent_variance)
        
        # 使用 decoder 生成重构概率
        reconstructed_probs = self.dec.decode(sampled_latent)
        
        # 计算重构损失，即负的重构概率的平均值
        reconstruction_losses = ut.log_bernoulli_with_logits(x, reconstructed_probs)
        reconstruction_loss = -torch.mean(reconstruction_losses)

        # 计算负的证据下界
        negative_elbo = kl_divergence + reconstruction_loss
        ################################################################################
        # 代码修改结束
        ################################################################################

        return negative_elbo, kl_divergence, reconstruction_loss

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        
        # 复制输入样本以生成多个重要性加权样本
        batch_size = x.shape[0]
        multi_x = ut.duplicate(x, num_samples)

        # 使用编码器对观测数据进行编码，得到潜在变量均值和方差
        q_mean, q_variance = self.enc.encode(x)
        multi_q_mean = ut.duplicate(q_mean, num_samples)
        multi_q_variance = ut.duplicate(q_variance, num_samples)

        # 使用重要性加权样本计算 IWAE 中的潜在变量 z
        z = ut.sample_gaussian(multi_q_mean, multi_q_variance)

        # 使用解码器生成每个样本对应的概率
        probabilities = self.dec.decode(z)

        # 计算负的重构损失，即负对数似然的平均值
        log_likelihoods = ut.log_bernoulli_with_logits(multi_x, probabilities)
        reconstruction_loss = -1.0 * torch.mean(log_likelihoods)

        # 准备计算 KL 散度的参数
        multi_prior_mean = self.z_prior[0].expand(multi_q_mean.shape)
        multi_prior_variance = self.z_prior[1].expand(multi_q_variance.shape)

        # 计算先验分布、后验分布和 KL 散度的对数概率
        prior_log_probs = ut.log_normal(z, multi_prior_mean, multi_prior_variance)
        posterior_log_probs = ut.log_normal(z, multi_q_mean, multi_q_variance)
        kl_divergence = torch.mean(prior_log_probs + log_likelihoods - posterior_log_probs)

        # 计算用于 IWAE 计算的 log ratios
        log_ratios = prior_log_probs + log_likelihoods - posterior_log_probs
        unflat_log_ratios = log_ratios.reshape(num_samples, batch_size)

        # 计算 IWAE 下界
        neg_iwae_bound = -1.0 * torch.mean(ut.log_mean_exp(unflat_log_ratios, 0))

        ################################################################################
        # 代码修改结束
        ################################################################################

        return neg_iwae_bound, kl_divergence, reconstruction_loss

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
