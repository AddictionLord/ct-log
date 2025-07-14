import plotly.express as px
from pathlib import Path

from src.dataset.ct_log_dataset import CTLogDataset


def main() -> None:
    """Visualize dataset images and masks, displaying and saving them as PNG files."""
    dataset = CTLogDataset(data_dir="data/processed/set_24", num_classes=10, resolution=(458, 530))

    output_dir = Path("data/processed/visualizations")
    output_dir.mkdir(exist_ok=True)

    print(f"Dataset length: {len(dataset)}")

    for i in range(len(dataset)):
        item = dataset[i]

        base_name = Path(item["path"]).stem

        img_fig = px.imshow(item["image"].permute(1, 2, 0), title=str(item["path"]))
        img_fig.show()
        img_fig.write_image(output_dir / f"{base_name}_image.png")

        mask_fig = px.imshow(item["mask"], title=str(item["path"]))
        mask_fig.show()
        mask_fig.write_image(output_dir / f"{base_name}_mask.png")

        # TODO: Create a function rendering the masks into the image.

        break


if __name__ == "__main__":
    main()
