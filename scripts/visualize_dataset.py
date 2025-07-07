import plotly.express as px

from src.dataset.ct_log_dataset import CTLogDataset


def main() -> None:
    dataset = CTLogDataset(data_dir="data/processed/set_24", num_classes=10)

    print(f"Dataset length: {len(dataset)}")

    for i in range(len(dataset)):
        item = dataset[i]

        px.imshow(item["mask"][0], title=str(item["path"])).show()

        break


if __name__ == "__main__":
    main()
