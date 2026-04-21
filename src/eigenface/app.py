from pca import train_eigenfaces, predict_face, nearest_neighbor
from dataset import load_dataset
import time



if __name__ == "__main__":
    folder_path="./data"
    heigth,width=64,64
    k=20
    iterations=10
    tolerance=1e-4

    start = time.time()

    print("Loading training images")
    test_matrix=load_dataset(folder_path,heigth,width)
    training_matrix=test_matrix[:-1]
    test_face=test_matrix[-1]
    print(f"Succesfully loaded {len(training_matrix)} amount of images")

    print("Training eigenfaces")
    mean, eigenvalues,eigenfaces,train_weights=train_eigenfaces(training_matrix,k,iterations,tolerance)
    print(f"Succesfully computed {len(eigenfaces)} top eigenfaces")


    print("Running test on test_image")
    test_weigths=predict_face(test_face,mean,eigenfaces)
    best_index,nearest_distance=nearest_neighbor(test_weigths,train_weights)
    end = time.time()
    print(f"Got result: index={best_index} and distance={nearest_distance}")

    print(f"Total runtime of the program is {end - start} seconds")