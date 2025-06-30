import plotly.express as px

from src.dataset.ct_log_mask_preprocessor import CTLogMaskPreprocessor


def main() -> None:
    dataset = CTLogMaskPreprocessor(data_dir="data/raw/set_24")

    print(f"Dataset length: {len(dataset)}")

    for i in range(len(dataset)):
        item = dataset[i]
        print(item["image"].shape)

        px.imshow(item["mask"], title=str(item["path"])).show()

        break


if __name__ == "__main__":
    main()
