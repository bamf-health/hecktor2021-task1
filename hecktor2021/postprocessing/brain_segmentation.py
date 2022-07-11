import numpy as np
import skimage


def remove_brain(pt: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """
    pt: PT scan
    predictions: predicted brain region
    pt and predictions are the same shape
    use the pt image to find the brain, then set all values in the brain region of predicitons to 0
    return the modified predictions array
    """
    threshold = np.mean(pt) + 3 * np.std(pt)  # set threshold to be 3x std dev.
    pt_thres = pt >= threshold  # convert pt to binary based on threshold

    labels = skimage.measure.label(
        pt_thres, background=0, connectivity=None
    )  # get labels
    props = skimage.measure.regionprops(labels)

    # get largest label
    sorted_labels = sorted(props, key=lambda x: x.area)
    largest_label = sorted_labels[-1].label

    brain = labels == largest_label

    # remove brain from AI prediction by appling brain label
    mod_predictions = np.copy(predictions)
    mod_predictions[brain] = 0

    return mod_predictions
