from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


class DataInjector(object):

    def __init__(self, tensor_datapack, sampler_type, batch_size):
        self.tensor_datapack = tensor_datapack
        self.sampler_type = sampler_type
        self.batch_size = batch_size

        self.dataset = self.define_dataset()
        self.sampler = self.define_sampler()
        self.dataloader = self.define_dataloader()

    def define_dataset(self):
        return TensorDataset(self.tensor_datapack.token_ids,
                             self.tensor_datapack.attention_mask,
                             self.tensor_datapack.labels)

    def define_sampler(self):
        if self.sampler_type == "random":
            return RandomSampler(self.dataset)
        if self.sampler_type == "sequential":
            return SequentialSampler(self.dataset)

    def define_dataloader(self):
        return DataLoader(self.dataset, sampler=self.sampler, batch_size=self.batch_size)