import os
from easydict import EasyDict
from dataset.tifo_dataset2 import TextToImageDataset, TextToImageDataloader


def make_cfg():
    cfg = EasyDict()
    cfg.model = EasyDict()
    cfg.model.processor_path = "/inspire/hdd/project/exploration-topic/public/downloaded_ckpts/Janus/Janus-Pro-7B"

    cfg.dataloader = EasyDict()
    cfg.dataloader.gen_data_path = "/inspire/hdd/project/exploration-topic/public/ent/download_dataset/BLIP3o-60k/*.tar"
    cfg.dataloader.num_workers = 0
    cfg.dataloader.prefetch_factor = 2
    cfg.dataloader.tasks = [0]

    cfg.dataloader.train = EasyDict()
    cfg.dataloader.train.task1 = EasyDict()
    cfg.dataloader.train.task1.batch_size = 8
    cfg.dataloader.train.task1.sample_ratio = 1

    cfg.dataloader.train.task2 = EasyDict()
    cfg.dataloader.train.task2.batch_size = 8
    cfg.dataloader.train.task2.sample_ratio = 0

    return cfg

def main():
    cfg = make_cfg()

    print("==== check dataset only ====")
    dataset = TextToImageDataset(
        model_path=cfg.model.processor_path,
        data_path=cfg.dataloader.gen_data_path,
    )
    print("dataset type:", type(dataset.dataset))
    print("dataset len:", len(dataset))

    sample = dataset[0]
    print("sample keys:", sample.keys())
    print("input_ids len:", len(sample["input_ids"]))
    print("image shape:", sample["image"].shape)
    print("task_type:", sample["task_type"])

    print("\n==== check dataloader function ====")
    loaders, probs = TextToImageDataloader(cfg, tasks=[0])
    print("num loaders:", len(loaders))
    print("probs:", probs)

    for i, loader in enumerate(loaders):
        print(f"loader[{i}] len =", len(loader))
        print(f"loader[{i}] sampler len =", len(loader.sampler))

if __name__ == "__main__":
    main()