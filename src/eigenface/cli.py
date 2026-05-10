"""Command line interface Eigenface algoritmin testaukseen.

Moduuli sisältää komennot mallin koulutukseen ja kasvojen tunnistukseen."""

import argparse
import time
import json
from pathlib import Path
#pylint: skip-file
from eigenface.dataset import load_dataset, load_image_as_vector
from eigenface.pca import train_eigenfaces, predict_face, nearest_neighbor
from eigenface.dataset import SUPPORTED_FILES
from PIL import Image, ImageDraw
import os
import subprocess
import tempfile


def build_parser():
    parser = argparse.ArgumentParser(prog="eigenface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train-command
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--k", type=int, default=20)
    train_parser.add_argument("--threshold", type=float, default=9.0)
    train_parser.add_argument("--dataset", default=str(Path(__file__).resolve().parent / "data"))
    train_parser.add_argument("--size", type=int, default=64)
    train_parser.add_argument("--iterations", type=int, default=10)
    train_parser.add_argument("--tolerance", type=float, default=1e-4)
    train_parser.add_argument("--model-out", default="src/eigenface/models/eigenface_model.json")

    # predict-command
    pred_parser = subparsers.add_parser("predict")
    pred_parser.add_argument("image")
    pred_parser.add_argument("--model", default="src/eigenface/models/eigenface_model.json")
    pred_parser.add_argument("--dataset", default=str(Path(__file__).resolve().parent / "data"))
    pred_parser.add_argument("--open", action="store_true")

    # evaluate-command
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--dataset", default=str(Path(__file__).resolve().parent / "data"))
    eval_parser.add_argument("--model", default="src/eigenface/models/eigenface_model.json")
    eval_parser.add_argument("--size", type=int, default=64)
    eval_parser.add_argument("--open", action="store_true")

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
        predict(args.image, args.model, dataset_path=args.dataset, open_images=args.open)

        return

    if args.command == "evaluate":
        print("Running evaluate")
        evaluate(
            dataset=args.dataset,
            model_path=args.model,
            size=args.size,
            open_images=args.open,
        )

        return

def train(folder_path, height, width, k, iterations, tolerance, threshold, model_out):
    start = time.time()

    print("Loading training images")
    matrix = load_dataset(folder_path, height, width)
    print(f"Succesfully loaded {len(matrix)} amount of training images")

    print("Training eigenfaces")
    mean, eigenvalues, eigenfaces, train_weights = train_eigenfaces(matrix, k, iterations, tolerance)

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





def collect_image_paths(dataset_path):
    paths = []

    for folder in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder)


        for file_name in sorted(os.listdir(folder_path)):
            image_path = os.path.join(folder_path, file_name)
            _, ext = os.path.splitext(file_name)

            if os.path.isfile(image_path) and ext.lower() in SUPPORTED_FILES:
                paths.append(image_path)

    return paths


def predict(image, model_path, dataset_path=None, open_images=False):
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
    best_index, nearest_distance = nearest_neighbor(test_weights, train_weights)

    if nearest_distance <= threshold:
        print("known face")
    else:
        print("not recognized")

    print(f"Got result: index={best_index} and distance={nearest_distance}")

    if not open_images:
        return

    if dataset_path is None:
        print("--open requires --dataset to be set")
        return

    train_images = collect_image_paths(dataset_path)
    nearest_image = train_images[best_index]

    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tf_path = tf.name
    tf.close()
    create_pair_image(image, nearest_image, width, tf_path)

    subprocess.Popen(["xdg-open", tf_path])




def evaluate(dataset, model_path, size, open_images=False):
    test_images, train_images = split_data(dataset)
    print(f"Selected {len(test_images)} held-out images and {len(train_images)} training images")

    train_matrix = []
    for image_path in train_images:
        train_matrix.append(load_image_as_vector(image_path, size, size))

    mean, eigenvalues, eigenfaces, train_weights = train_eigenfaces(
        train_matrix, k=min(20, len(train_matrix)), iterations=10, tolerance=1e-4
    )

    model_data = {
        "height": size,
        "width": size,
        "k": min(20, len(train_matrix)),
        "iterations": 10,
        "tolerance": 1e-4,
        "mean": mean,
        "eigenvalues": eigenvalues,
        "eigenfaces": eigenfaces,
        "train_weights": train_weights,
    }

    with open(model_path, "w", encoding="utf-8") as model_file:
        json.dump(model_data, model_file)

    # käydään läpi testikuvat ja etsitään lähin koulutus kuva
    amount = 0
    for img_path in test_images:
        v = load_image_as_vector(img_path, size, size)
        w = predict_face(v, mean, eigenfaces)

        best_index, dist = nearest_neighbor(w, train_weights)
        ref_path = train_images[best_index]

        # temporrary kuvan luonti
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tf_path = tf.name
        tf.close()
        create_pair_image(img_path, ref_path, size, tf_path)
        print(f"Generated temporary picture: {tf_path} distance={dist}")

        if open_images:
                subprocess.Popen(["xdg-open", tf_path])
        amount += 1

    print(f"Processed {amount} test images")

def image_sort_key(filename):
    stem = Path(filename).stem
    if stem.isdigit():
        return (0, int(stem))
    else:
        return (1, stem)


def split_data(dataset_path):
    test_images = []
    train_images = []

    for entry in sorted(os.listdir(dataset_path)):
        folder = os.path.join(dataset_path, entry)
        if not os.path.isdir(folder):
            continue

        image_paths = []

        for fname in os.listdir(folder):
            _, ext = os.path.splitext(fname)
            if ext.lower() in SUPPORTED_FILES:
                image_paths.append(os.path.join(folder, fname))

        image_paths.sort(key=image_sort_key)
        if not image_paths:
            continue

        test_images.append(image_paths[0])
        train_images.extend(image_paths[1:])

    return test_images, train_images

# laitetaan kuva ja ennustettu kuva vierekkäin
def create_pair_image(img_a_path, img_b_path, size, out_path):
    with Image.open(img_a_path) as a, Image.open(img_b_path) as b:
        a = a.convert("L").resize((size, size))
        b = b.convert("L").resize((size, size))

        text_heigth=20
        out_img = Image.new("L", (size * 2, size+text_heigth), color=255)
        out_img.paste(a, (0, text_heigth))
        out_img.paste(b, (size, text_heigth))

        draw = ImageDraw.Draw(out_img)
        draw.text((10, 0), "Test picture", fill=0)
        draw.text((size + 10, 0), "Most similar", fill=0)

        out_img.save(out_path)



if __name__ == "__main__":
    main()
