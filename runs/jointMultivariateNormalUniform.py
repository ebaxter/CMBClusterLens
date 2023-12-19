import torch
import pdb

#Create joint distribution combining independent multivariate normal and uniform distributions
# Expects normal distribution first, then uniform distribution
class jointMultivariateNormalUniform(torch.distributions.Distribution):
    def __init__(self, normal_dist, uniform_dist):
        self.normal_dist = normal_dist
        self.uniform_dist = uniform_dist
        #Event shape needs to match length of pvector
        self.dim_normal = normal_dist.event_shape[0]
        self.dim_uniform = uniform_dist.event_shape[0]

    def log_prob(self, value):
        # Distributions are treated as independent, so just add log probs
        normal_log_prob = self.normal_dist.log_prob(value[:,0:self.dim_normal])
        uniform_log_prob = self.uniform_dist.log_prob(value[:,self.dim_normal:])
        return normal_log_prob + uniform_log_prob

    def sample(self, sample_shape=torch.Size()):
        #output will have dimension of sample_shape
        #sample separately from each distribution, and then concatenate
        normal_sample = self.normal_dist.sample(sample_shape)  # this will be sample_shape x dim_normal
        uniform_sample = self.uniform_dist.sample(sample_shape) # this will be sample_shape x dim_uniform
        #concatenate along last dimension
        joint_sample = torch.cat((normal_sample, uniform_sample), dim = -1)
        return joint_sample

    def rsample(self, sample_shape=torch.Size()):
        normal_sample = self.normal_dist.rsample(sample_shape)
        uniform_sample = self.uniform_dist.rsample(sample_shape)
        joint_sample = torch.cat((normal_sample, uniform_sample), dim = -1)
        return joint_sample

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(jointMultivariateNormalUniform, _instance)
        batch_shape = torch.Size(batch_shape)
        new.normal_dist = self.normal_dist.expand(batch_shape, _instance)
        new.uniform_dist = self.uniform_dist.expand(batch_shape, _instance)
        super(jointMultivariateNormalUniform, new).__init__(batch_shape, new.normal_dist.event_shape, _validate_args=False)
        new._validate_args = self._validate_args
        return new
    @property
    def mean(self):
        return torch.cat((self.normal_dist.mean, self.uniform_dist.mean), dim = -1)

    @property
    def variance(self):
        return torch.cat((self.normal_dist.variance, self.uniform_dist.variance), dim = -1)

    @property
    def stddev(self):
        return torch.cat((self.normal_dist.stddev, self.uniform_dist.stddev), dim = -1)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'normal_dist=' + str(self.normal_dist) + ', uniform_dist=' + str(self.uniform_dist) + ')'

    def __len__(self):
        return len(self.normal_dist)

    #def __getitem__(self, key):
    #    return jointMultivariateNormalUniform(self.normal_dist[key], self.uniform_dist[key])




    