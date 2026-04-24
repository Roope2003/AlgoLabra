from pca import train_eigenfaces, predict_face, nearest_neighbor
from dataset import load_dataset
import time



if __name__ == "__main__":
    folder_path="./data"
    heigth,width=64,64
    k=20
    iterations=10
    tolerance=1e-4
    threshold=9
    distances=[]
    start = time.time()

    print("Loading training images")
    test_matrix,train_matrix=load_dataset(folder_path,heigth,width)
    training_matrix=train_matrix
    print(f"Succesfully loaded {len(training_matrix)} amount of training images and {len(test_matrix)} of testing images")

    print("Training eigenfaces")
    mean, eigenvalues,eigenfaces,train_weights=train_eigenfaces(training_matrix,k,iterations,tolerance)
    print(f"Succesfully computed {len(eigenfaces)} top eigenfaces")


    print("Running tests on test_matrix")
    for test_face in test_matrix:
        test_weigths=predict_face(test_face,mean,eigenfaces)
        best_index,nearest_distance=nearest_neighbor(test_weigths,train_weights)
        print(f"Got result: index={best_index} and distance={nearest_distance}")
        distances.append(nearest_distance)
        if nearest_distance<= threshold:
            print("known face")
        else:
            print("not recognized")

    end = time.time()

    print(f"Total runtime of the program is {end - start} seconds")
    print(distances)
    avg=sum(distances)/len(distances)
    print((f"Average={avg}, min={min(distances)},max={max(distances)}"))