from concurrent.futures import ProcessPoolExecutor
import os
import cv2 as cv
import numpy as np
import sys
import time

# Global variable for manually selected points
ročno_izbrane_točke = []


def klik_na_sliko(event, x, y, flags, param):
    """Mouse callback to collect clicked points."""
    global ročno_izbrane_točke
    if event == cv.EVENT_LBUTTONDOWN:
        ročno_izbrane_točke.append((y, x))  # Store (row, col)


def izracunaj_centre(slika, izbira="nakljucna", dimenzija_centra=3, T=0.3, k=3):
    """
    This function initializes the cluster centers for K-means.
    Depending on the mode (random or manual), it selects `k` starting centers
    from the feature space, either based on color or color+position.
    """
    h, w, _ = slika.shape

    # Build the feature space based on chosen dimensionality
    if dimenzija_centra == 3:
        # Use only RGB color values (normalized)
        features = slika.reshape(-1, 3).astype(np.float32) / 255.0
    elif dimenzija_centra == 5:
        # Use RGB + spatial coordinates (normalized XY)
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        coords = np.stack((yy, xx), axis=-1).reshape(-1, 2).astype(np.float32)
        coords[:, 0] /= h
        coords[:, 1] /= w
        colors = slika.reshape(-1, 3).astype(np.float32) / 255.0
        features = np.hstack((colors, coords))
    else:
        raise ValueError("dimenzija_centra must be 3 or 5")

    # If mode is random, pick diverse initial centers
    if izbira == "nakljucna":
        centri = []
        attempts = 0
        max_attempts = 1000  # Avoid infinite loop on low-diversity images

        while len(centri) < k and attempts < max_attempts:
            kandidat = features[np.random.randint(len(features))]

            # Manhattan distance used to avoid selecting very similar centers
            if all(np.linalg.norm(kandidat - c, ord=1) > T for c in centri):
                centri.append(kandidat)
            attempts += 1

        if len(centri) < k:
            print(
                f"[WARN] Only {len(centri)} centers found after {max_attempts} attempts."
            )

        return np.array(centri)

    # If mode is manual, user clicks on the image
    elif izbira == "ročno":
        global ročno_izbrane_točke
        ročno_izbrane_točke = []

        # Set up OpenCV window and collect k clicks
        slika_kopija = slika.copy()
        cv.imshow("Click to select centers (ESC to confirm)", slika_kopija)
        cv.setMouseCallback("Click to select centers (ESC to confirm)", klik_na_sliko)

        print(f"[INFO] Click {k} points on the image.")
        while True:
            cv.imshow("Click to select centers (ESC to confirm)", slika_kopija)
            if len(ročno_izbrane_točke) >= k:
                break
            key = cv.waitKey(20)
            if key == 27:  # ESC
                break

        cv.destroyAllWindows()

        # Convert clicked coordinates into feature vectors
        centri = []
        for y, x in ročno_izbrane_točke[:k]:
            barva = slika[y, x].astype(np.float32) / 255.0
            if dimenzija_centra == 3:
                centri.append(barva)
            else:
                centri.append(np.hstack((barva, [y / h, x / w])))

        return np.array(centri)

    else:
        raise ValueError("Invalid 'izbira': must be 'nakljucna' or 'ročno'")


def kmeans(slika, k=3, iteracije=10, dimenzija_centra=3, izbira="nakljucna", T=0.3):
    """
    Segments the input image using the K-means clustering algorithm.
    Each pixel is treated as a point in 3D (RGB) or 5D (RGB + XY) feature space.
    Manhattan distance (L1) is used for pixel-to-center assignment.
    """

    h, w, _ = slika.shape

    # Build the full feature space for all pixels
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = np.stack((yy, xx), axis=-1).reshape(-1, 2).astype(np.float32)
    coords[:, 0] /= h
    coords[:, 1] /= w
    colors = slika.reshape(-1, 3).astype(np.float32) / 255.0
    features = np.hstack((colors, coords)) if dimenzija_centra == 5 else colors

    # Get initial centers using the helper function
    centri = izracunaj_centre(
        slika, izbira=izbira, dimenzija_centra=dimenzija_centra, T=T, k=k
    )
    k = len(centri)  # Adjust in case fewer centers were selected

    for _ in range(iteracije):
        # Compute Manhattan distances between all pixels and all centers
        dists = np.sum(np.abs(features[:, None] - centri[None, :]), axis=2)

        # Assign each pixel to the nearest cluster
        labels = np.argmin(dists, axis=1)

        # Update each center to be the mean of its assigned points
        for i in range(k):
            pripadajoči = features[labels == i]
            if len(pripadajoči) > 0:
                centri[i] = np.mean(pripadajoči, axis=0)

    # Build final segmented image using the first 3 values (RGB) of each center
    rezultat = np.zeros((h * w, 3), dtype=np.uint8)
    for i in range(k):
        rezultat[labels == i] = (centri[i][:3] * 255).astype(np.uint8)

    return rezultat.reshape((h, w, 3)), labels


def kmeans_example(input_image, k=4, iteracije=10, dimenzija_centra=5):
    """
    Runs K-means on the provided image and returns a color-visualized result.
    Each cluster is assigned a unique display color for easy visual verification.
    Requires kmeans() to return both (segmented_img, labels).
    """
    h, w = input_image.shape[:2]

    # Run updated kmeans that returns (segmented_image, labels)
    segmented_img, labels = kmeans(
        input_image, k=k, iteracije=iteracije, dimenzija_centra=dimenzija_centra
    )

    # Assign fixed artificial colors for visual separation
    display_colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 128, 0),
        (128, 0, 128),
    ]
    visual = np.zeros((h * w, 3), dtype=np.uint8)
    for i in range(k):
        visual[labels == i] = display_colors[i % len(display_colors)]

    return visual.reshape((h, w, 3))


def shift_chunk(indices, shifted, velikost_okna):
    """
    Shift a subset of points based on their local neighborhood.

    This function implements the core Mean-Shift equation:
        m(x_i) = sum(K(||x_i - x_j||^2, h) * x_j) / sum(K(...))
    where:
        - K(d^2, h) = exp(-d^2 / (2 * h^2)) is the Gaussian kernel
        - x_j are neighbors within window size `h` of x_i
    """
    chunk_result = []

    for i in indices:
        xi = shifted[i]  # Current point
        dists = np.linalg.norm(shifted - xi, axis=1)  # Distances to all others

        # Neighborhood mask: include only nearby points
        mask = dists < velikost_okna
        if not np.any(mask):
            chunk_result.append(xi)
            continue

        # Compute weights using the Gaussian kernel
        weights = np.exp(-(dists[mask] ** 2) / (2 * velikost_okna**2))

        # Compute the weighted mean of neighbors
        new_point = np.average(shifted[mask], axis=0, weights=weights)
        chunk_result.append(new_point)

    return chunk_result


def chunk_indices(n, num_chunks):
    """Split list of indices [0, ..., n-1] into num_chunks parts."""
    chunk_size = (n + num_chunks - 1) // num_chunks
    return [list(range(i, min(i + chunk_size, n))) for i in range(0, n, chunk_size)]


def meanshift(slika, velikost_okna, dimenzija, max_iter=10, min_cd=0.05):
    """
    Perform Mean-Shift segmentation on an image.

    Each pixel is treated as a point in 3D (color) or 5D (color + position) space.
    Points iteratively shift toward the weighted mean of their neighbors using
    a Gaussian kernel. The final image is formed by merging nearby converged points.
    """
    h, w, _ = slika.shape
    # max_iter = 10  # Maximum number of shifts per point
    # min_cd = 0.05  # Minimum distance between centers (in normalized space)

    # Build the feature space: either only color or color + position
    if dimenzija == 3:
        features = slika.reshape(-1, 3).astype(np.float32) / 255.0
    elif dimenzija == 5:
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        coords = np.stack((yy, xx), axis=-1).reshape(-1, 2).astype(np.float32)
        coords[:, 0] /= h  # Normalize coordinates to [0, 1]
        coords[:, 1] /= w
        colors = slika.reshape(-1, 3).astype(np.float32) / 255.0
        features = np.hstack((colors, coords))
    else:
        raise ValueError("dimenzija must be 3 or 5")

    shifted = features.copy()  # Initialize shifted points with original features

    # Determine number of parallel worker processes
    num_workers = os.cpu_count()
    print(f"Num workers: {num_workers}")
    index_chunks = chunk_indices(len(shifted), num_workers)

    # Perform Mean-Shift iterations using parallel processing
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for it in range(max_iter):
                print(f"[INFO] Starting iteration {it + 1}/{max_iter}...")
                start_time = time.time()

                # Each worker shifts a chunk of points
                futures = [
                    executor.submit(shift_chunk, chunk, shifted, velikost_okna)
                    for chunk in index_chunks
                ]
                results = []
                for f in futures:
                    results.extend(f.result())
                shifted = np.array(results, dtype=np.float32)

                duration = time.time() - start_time
                print(f"[INFO] Iteration {it + 1} took {duration:.2f} seconds")

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user. Shutting down workers...")
        executor.shutdown(wait=False, cancel_futures=True)
        os._exit(1)

    # After convergence, merge similar shifted points into unique centers
    centri = []
    labels = np.zeros(len(features), dtype=np.int32)
    for i, p in enumerate(shifted):
        for j, c in enumerate(centri):
            if np.linalg.norm(p - c) < min_cd:
                labels[i] = j
                break
        else:
            centri.append(p)
            labels[i] = len(centri) - 1

    # Build final segmented image using the first 3 components (RGB) of centers
    rezultat = np.zeros((len(features), 3), dtype=np.uint8)
    for i, c in enumerate(centri):
        rezultat[labels == i] = (c[:3] * 255).astype(np.uint8)

    return rezultat.reshape((h, w, 3))


def run_report_generation(slika):
    os.makedirs("report", exist_ok=True)

    # --- K-MEANS TESTS ---
    kmeans_tests = [
        {
            "name": "kmeans_rgb_random_T03",
            "args": {
                "k": 6,
                "iteracije": 10,
                "dimenzija_centra": 3,
                "izbira": "nakljucna",
                "T": 0.3,
            },
        },
        {
            "name": "kmeans_rgb_pos_random_T03",
            "args": {
                "k": 6,
                "iteracije": 10,
                "dimenzija_centra": 5,
                "izbira": "nakljucna",
                "T": 0.3,
            },
        },
        {
            "name": "kmeans_rgb_pos_random_T01",
            "args": {
                "k": 6,
                "iteracije": 10,
                "dimenzija_centra": 5,
                "izbira": "nakljucna",
                "T": 0.01,
            },
        },
        {
            "name": "kmeans_rgb_random_T06",
            "args": {
                "k": 6,
                "iteracije": 10,
                "dimenzija_centra": 3,
                "izbira": "nakljucna",
                "T": 0.6,
            },
        },
    ]

    for test in kmeans_tests:
        print(f"[INFO] Running {test['name']}...")
        img, _ = kmeans(slika, **test["args"])
        cv.imwrite(f"report/{test['name']}.png", img)
    """
    # --- MEAN-SHIFT TESTS ---
    meanshift_tests = [
        {"name": "meanshift_rgb_01", "args": {"velikost_okna": 0.1, "dimenzija": 3}},
        {
            "name": "meanshift_rgb_pos_01",
            "args": {"velikost_okna": 0.1, "dimenzija": 5},
        },
        {
            "name": "meanshift_rgb_pos_005",
            "args": {"velikost_okna": 0.05, "dimenzija": 5},
        },
        {
            "name": "meanshift_rgb_pos_02",
            "args": {"velikost_okna": 0.2, "dimenzija": 5},
        },
        {
            "name": "meanshift_rgb_pos_01_min01",
            "args": {"velikost_okna": 0.1, "dimenzija": 5, "min_cd": 0.01},
        },
    ]

    for test in meanshift_tests:
        print(f"[INFO] Running {test['name']}... (downscaled)")
        downscale_factor = 0.25
        new_size = (
            int(slika.shape[1] * downscale_factor),
            int(slika.shape[0] * downscale_factor),
        )
        small_img = cv.resize(slika, new_size, interpolation=cv.INTER_AREA)

        result_small = meanshift(small_img, **test["args"])
        result = cv.resize(
            result_small,
            (slika.shape[1], slika.shape[0]),
            interpolation=cv.INTER_NEAREST,
        )

        cv.imwrite(f"report/{test['name']}.png", result)
    """
    print("[INFO] All report images saved in 'report/' folder.")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")  # optional, but explicit

    if len(sys.argv) < 2 or sys.argv[1] not in ["km", "ms", "mss", "kme", "report"]:
        print("Usage: python naloga3.py [km|ms]")
        sys.exit(1)

    slika = cv.imread(".utils/zelenjava.jpg")
    if slika is None:
        raise FileNotFoundError("Image not found. Check the path to zelenjava.jpg")

    if sys.argv[1] == "km":
        test1 = cv.imread("test_image_diff_reds.png")
        print("[INFO] Running K-means segmentation...")
        rezultat_kmeans = kmeans(test1, k=3, iteracije=10, dimenzija_centra=3)
        cv.imwrite("rezultat_kmeans.png", rezultat_kmeans)
        print("Saved: rezultat_kmeans.png")

    elif sys.argv[1] == "ms":
        print("[INFO] Running Mean-Shift segmentation...")
        rezultat_meanshift = meanshift(slika, velikost_okna=0.1, dimenzija=3)
        cv.imwrite("rezultat_meanshift.png", rezultat_meanshift)
        print("Saved: rezultat_meanshift.png")

    elif sys.argv[1] == "mss":
        print("[INFO] Running Mean-Shift segmentation (downscaled)...")

        downscale_factor = 0.25  # try 0.25 or 0.5 for testing
        original_size = (slika.shape[1], slika.shape[0])
        new_size = (
            int(original_size[0] * downscale_factor),
            int(original_size[1] * downscale_factor),
        )
        slika_small = cv.resize(slika, new_size, interpolation=cv.INTER_AREA)

        rezultat_meanshift_small = meanshift(
            slika_small, velikost_okna=0.1, dimenzija=3
        )

        rezultat_meanshift = cv.resize(
            rezultat_meanshift_small, original_size, interpolation=cv.INTER_NEAREST
        )

        cv.imwrite("rezultat_meanshift.png", rezultat_meanshift)
        print("Saved: rezultat_meanshift.png")
    elif sys.argv[1] == "kme":
        print("[INFO] Running K-means example with synthetic red squares...")

        test_img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv.rectangle(test_img, (20, 20), (90, 90), (0, 0, 255), -1)
        cv.rectangle(test_img, (120, 120), (160, 160), (0, 0, 200), -1)
        cv.rectangle(test_img, (165, 130), (190, 155), (0, 0, 200), -1)

        visual = kmeans_example(test_img, k=4, iteracije=10, dimenzija_centra=3)

        cv.imwrite("kmeans_test_input.png", test_img)
        cv.imwrite("kmeans_test_result_visual.png", visual)

        print("Saved: kmeans_test_input.png")
        print("Saved: kmeans_test_result_visual.png")
    elif sys.argv[1] == "report":
        run_report_generation(slika)
