from os import makedirs, path

from numpy import ndarray
from pypylon import pylon


def connect_camera(buffer_val: int, expre: int) -> pylon.InstantCamera:
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.MaxNumBuffer.Value = buffer_val
    camera.ExpreTime.Value = expre
    return camera


def return_single_image(camera: pylon.InstantCamera) -> ndarray:
    camera.StartGrabbingMax(1)

    with camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException) as result:
        if result.GrabSucceeded():
            image_array = result.GetArray()
        else:
            raise RuntimeError("Image grab failed")

    camera.StopGrabbing()

    return image_array


def save_images(camera: pylon.InstantCamera, num: int, file_path: str) -> None:
    makedirs(file_path, exist_ok=True)

    camera.StartGrabbingMax(num)

    grab_idx = 0
    while camera.IsGrabbing():
        with camera.RetrieveResult(2000) as result:
            if result.GrabSucceeded():
                img = pylon.PylonImage()
                img.AttachGrabResultBuffer(result)
                filename = path.join(file_path, f"{grab_idx}.tiff")
                img.Save(pylon.ImageFileFormat_Tiff, filename)
                img.Release()
                print(f"Saved image: {filename}")
                grab_idx += 1  # Increment grab index
            else:
                print(f"Grab failed with error: {result.ErrorCode}")

    camera.StopGrabbing()
