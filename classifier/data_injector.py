from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from typing import Union

from classifier.tensor_datapack import TensorDatapack


class DataInjector(object):

    def __init__(self, tensor_datapack: TensorDatapack, sampler_type, batch_size: int):
        """
        Class initializer.
        Args:
            tensor_datapack: Object with tensors.
            sampler_type: String indicating the sampler type: "random" or "sequential".
            batch_size: Batch size for the training loop.
        """
        self.tensor_datapack = tensor_datapack
        self.sampler_type = sampler_type
        self.batch_size = batch_size

        self.dataset = self._define_dataset()
        self.sampler = self._define_sampler()
        self.dataloader = self._define_dataloader()

    def _define_dataset(self) -> TensorDataset:
        """
        Creates TensorDataset with the given token ids, attention mask and labels.
        Returns:
            TensorDataset for the given data.
        """
        return TensorDataset(self.tensor_datapack.token_ids,
                             self.tensor_datapack.attention_mask,
                             self.tensor_datapack.labels)

    def _define_sampler(self) -> Union[RandomSampler, SequentialSampler]:
        """
        Creates a sampler based on the defined strategy.
        Returns:
            Sampler object (RandomSampler or SequentialSampler) based on the strategy selected.
        """
        if self.sampler_type == "random":
            return RandomSampler(self.dataset)
        if self.sampler_type == "sequential":
            return SequentialSampler(self.dataset)

    def _define_dataloader(self) -> DataLoader:
        """
        Creates DataLoader with the given dataset, sampler type and batch size.
        Returns:
            DataLoader for the given data.
        """
        return DataLoader(self.dataset, sampler=self.sampler, batch_size=self.batch_size)
