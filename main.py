from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import numpy as np
import cv2
Here are some improvements to the given Python program:

1. Move import statements to the top of the file and group them together for better readability.
2. Use type hints in the method signatures to improve clarity and help with IDE autocompletion.
3. Remove unnecessary import of the `numpy` module since it is already imported as `np`.
4. Add docstrings to the class and methods to provide explanations.
5. Rename the `_extract_features` method to `extract_features` since it is not a private method.
6. Convert the list of training and test images to numpy arrays to improve performance.
7. Remove unnecessary conversion of RGB images to BGR using `cv2.cvtColor` since the given example assumes the images are already in RGB format.
8. Use list comprehension to simplify the creation of the recommendations list.
9. Use the `enumerate` function to iterate over the recommendations and get the index and item directly.

Here is the improved code:

```python


class ImageRecognitionSystem:
    def __init__(self, num_clusters: int = 10, num_recommendations: int = 5) -> None:
        """
        Initialize the ImageRecognitionSystem.

        Args:
            num_clusters: Number of clusters for K-means clustering.
            num_recommendations: Number of recommendations to return.
        """
        self.num_clusters = num_clusters
        self.num_recommendations = num_recommendations
        self.kmeans = None
        self.nearest_neighbors = None

    def train(self, images: np.ndarray) -> None:
        """
        Train the ImageRecognitionSystem.

        Args:
            images: Array of training images.
        """
        features = self.extract_features(images)
        self.kmeans = KMeans(n_clusters=self.num_clusters,
                             random_state=0).fit(features)
        self.nearest_neighbors = NearestNeighbors(
            metric="euclidean").fit(features)

    def recommend(self, image: np.ndarray) -> List[Tuple[float, int]]:
        """
        Recommend similar images based on the given image.

        Args:
            image: Test image for recommendation.

        Returns:
            List of recommendations as tuples: (distance, index).
        """
        if self.kmeans is None or self.nearest_neighbors is None:
            raise RuntimeError("The system has not been trained yet.")

        query_features = self.extract_features(np.array([image]))
        query_cluster = self.kmeans.predict(query_features)
        cluster_indices = np.where(self.kmeans.labels_ == query_cluster)[0]

        distances, indices = self.nearest_neighbors.kneighbors(
            query_features, n_neighbors=self.num_recommendations, indices=cluster_indices
        )

        recommendations = [(distances[0][i], indices[0][i])
                           for i in range(len(indices[0]))]
        recommendations = sorted(recommendations, key=lambda x: x[0])

        return recommendations

    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract features from the given images.

        Args:
            images: Array of images.

        Returns:
            Array of feature vectors.
        """
        features = [image.flatten() for image in images]
        features = np.array(features)
        features = normalize(features)

        return features


# Example usage
if __name__ == "__main__":
    image_recognition = ImageRecognitionSystem(
        num_clusters=10, num_recommendations=5)

    # Training phase
    training_images = np.array([...])  # Array of training images
    image_recognition.train(training_images)

    # Recommendation phase
    test_image = np.array([...])  # Test image for recommendation
    recommendations = image_recognition.recommend(test_image)

    for i, (distance, index) in enumerate(recommendations):
        print(
            f"Recommendation {i + 1}: Distance = {distance}, Index = {index}")
```

These improvements should help make the code more efficient, readable, and maintainable.
