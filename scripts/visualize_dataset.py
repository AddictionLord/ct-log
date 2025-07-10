import plotly.express as px

from src.dataset.ct_log_dataset import CTLogDataset


def main() -> None:
    dataset = CTLogDataset(data_dir="data/processed/set_24", num_classes=10, resolution=(458, 530))

    print(f"Dataset length: {len(dataset)}")

    for i in range(len(dataset)):
        item = dataset[i]

        px.imshow(item["image"].permute(1, 2, 0), title=str(item["path"])).show()
        px.imshow(item["mask"], title=str(item["path"])).show()

        break


if __name__ == "__main__":
    main()
