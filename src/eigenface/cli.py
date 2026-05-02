"""Command line interface Eigenface algoritmin testaukseen.

Moduuli sisältää komennot mallin koulutukseen ja kasvojen tunnistukseen."""



import argparse
import time
import json
from pathlib import Path
#pylint: skip-file
from eigenface.dataset import load_dataset, load_image_as_vector
from eigenface.pca import train_eigenfaces, predict_face, nearest_neighbor


def build_parser():
    parser = argparse.ArgumentParser(prog="eigenface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train-command
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--k", type=int, default=20)
    train_parser.add_argument("--threshold", type=float, default=9.0)
    train_parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parent / "data"),
    )

    train_parser.add_argument("--size", type=int, default=64)
    train_parser.add_argument("--iterations", type=int, default=10)
    train_parser.add_argument("--tolerance", type=float, default=1e-4)
    train_parser.add_argument("--model-out", default="eigenface/models/eigenface_model.json")

    # predict-command
    pred_parser = subparsers.add_parser("predict")
    pred_parser.add_argument("image")
    pred_parser.add_argument("--model", default="eigenface/models/eigenface_model.json")

    return parser



def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        print("Running train")
        train(
            folder_path=args.dataset,
            height=args.size,
            width=args.size,
            k=args.k,
            iterations=args.iterations,
            tolerance=args.tolerance,
            threshold=args.threshold,
            model_out=args.model_out,
        )
        return

    if args.command == "predict":
        print("Running predict")
        predict(args.image, args.model)
        return


def train(folder_path, height, width, k, iterations, tolerance, threshold, model_out):
    start = time.time()

    print("Loading training images")
    matrix = load_dataset(folder_path, height, width)
    print(
        f"Succesfully loaded {len(matrix)} amount of training images and ")

    print("Training eigenfaces")
    mean, eigenvalues, eigenfaces, train_weights = train_eigenfaces(
        matrix, k, iterations, tolerance
    )
    print(f"Succesfully computed {len(eigenfaces)} top eigenfaces")

    model_data = {
        "height": height,
        "width": width,
        "k": k,
        "iterations": iterations,
        "tolerance": tolerance,
        "threshold": threshold,
        "mean": mean,
        "eigenvalues": eigenvalues,
        "eigenfaces": eigenfaces,
        "train_weights": train_weights,
    }

    with open(model_out, "w", encoding="utf-8") as m_out:
        json.dump(model_data, m_out)

    end = time.time()

    print(f"Total runtime of the program is {end - start} seconds")
    print(f"Model saved to: {model_out}")


def predict(image, model_path):

    with open(model_path, "r", encoding="utf-8") as model_file:
        model_data = json.load(model_file)

    mean = model_data["mean"]
    eigenfaces = model_data["eigenfaces"]
    train_weights = model_data["train_weights"]
    height = model_data["height"]
    width = model_data["width"]
    threshold = model_data.get("threshold", 9.0)

    print("Running test on image")
    image_v = load_image_as_vector(image, height, width)
    test_weights = predict_face(image_v, mean, eigenfaces)

    best_index, nearest_distance = nearest_neighbor(
        test_weights, train_weights
    )

    if nearest_distance <= threshold:
        print("known face")
    else:
        print("not recognized")

    print(f"Got result: index={best_index} and distance={nearest_distance}")









if __name__ == "__main__":
    main()
