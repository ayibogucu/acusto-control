import os

from pipython import pitools
from pypylon import genicam
from ultralytics import YOLO

import lib.cmr as cmr
import lib.cnf as cnf
import lib.mtr as mtr
from lib.cell_detection import get_bounding_boxes

import lib.fcs as fcs


def main():
    print("Loading configuration...")
    config = cnf.load_config("config.toml")

    print("Connecting to the motor controller...")
    pidevice = mtr.connect_pi(
        config.motor.controllername,
        config.motor.serialnum,
        config.motor.stages,
        config.motor.refmodes,
    )

    print("Connecting to the camera...")
    camera = cmr.connect_camera(500, config.camera.exposure, config.camera.fps)
    os.makedirs(config.file.save_dir, exist_ok=True)

    print("Loading the yolov8 model...")
    model = YOLO(config.file.model_path)

    try:
        offset_x_increment = camera.OffsetX.GetInc()
        offset_y_increment = camera.OffsetY.GetInc()
        width_increment = camera.Width.GetInc()
        height_increment = camera.Height.GetInc()
        max_width = camera.Width.GetMax()
        max_height = camera.Height.GetMax()

        print(
            f"Width Increment: {width_increment}, Height Increment: {height_increment}"
        )
        print(
            f"OffsetX Increment: {offset_x_increment}, OffsetY Increment: {offset_y_increment}"
        )

        print(f"Maximum Width: {max_width}, Maximum Height: {max_height}")

    except genicam.GenericException as e:
        print(f"Error retrieving increment and maximum values: {e}")
        camera.Close()
        pidevice.CloseConnection()
        return

    print("Starting scanning process...")

    for y in range(config.movement.num_steps_y + 1):
        for x in (
            range(config.movement.num_steps_x + 1)
            if y % 2 == 0
            else range(config.movement.num_steps_x + 1)[::-1]
        ):
            target_x = config.vertex.pt1[0] + x * config.movement.dx
            target_y = config.vertex.pt1[1] + y * config.movement.dy
            print(f"\nMoving to position: X={target_x}, Y={target_y}")

            try:
                pidevice.MOV([config.axes.x, config.axes.y], [target_x, target_y])
                pitools.waitontarget(pidevice, axes=(config.axes.x, config.axes.y))
                print("Stage movement complete.")
            except Exception as e:
                print(f"Error during stage movement: {e}")
                continue

            try:
                print("Adjusting focus...")
                fcs.move_to_focus(pidevice, camera, config)
                print("Focus has been achieved.")
            except Exception as e:
                print(f"Error during focusing: {e}")
                continue

            print("Capturing original image...")
            temp_dir = os.path.join(config.file.save_dir, "temp")
            try:
                cmr.save_images(camera, 1, temp_dir)
            except Exception as e:
                print(f"Error capturing image: {e}")
                continue

            print("Detecting circles...")
            temp_file = os.path.join(temp_dir, "0.tiff")
            bboxes = get_bounding_boxes(model, temp_file, 40)

            if bboxes is None:
                print("There are no circles detected. Skipping current position...")
                continue

            print(f"Detected {len(bboxes)} circles.")

            for idx, bbox in enumerate(bboxes):
                frame_dir = os.path.join(
                    config.file.save_dir,
                    f"position({target_x:.2f},{target_y:.2f})_cell{idx}",
                )

                x_min, y_min, x_max, y_max = bbox
                print(
                    f"\nProcessing Circle {idx}: top left corner ({x_min}, {y_min}), bottom right corner({x_max}, {y_max})"
                )

                try:
                    # NOTE: if you want a specific size uncomment this and comment the other paragraph
                    # camera.Width.Value = config.camera.kernel_size[0]
                    # camera.Height.Value = config.camera.kernel_size[1]

                    adjusted_width = adjust(
                        abs(x_max - x_min), width_increment, max_width
                    )
                    adjusted_height = adjust(
                        abs(y_max - y_min), height_increment, max_height
                    )

                    camera.Width.Value = adjusted_width
                    camera.Height.Value = adjusted_height

                    current_max_offset_x = camera.OffsetX.GetMax()
                    current_max_offset_y = camera.OffsetY.GetMax()

                    adjusted_offset_x = adjust(
                        x_min, offset_x_increment, current_max_offset_x
                    )
                    adjusted_offset_y = adjust(
                        y_min, offset_y_increment, current_max_offset_y
                    )

                    camera.OffsetX.Value = adjusted_offset_x
                    camera.OffsetY.Value = adjusted_offset_y

                    print(
                        f"Adjusted Camera ROI: Width={camera.Width.Value}, Height={camera.Height.Value}, "
                        f"OffsetX={camera.OffsetX.Value}, OffsetY={camera.OffsetY.Value}"
                    )

                    try:
                        print("Adjusting focus...")
                        fcs.move_to_focus(pidevice, camera, config)
                        print("Focus has been achieved.")
                    except Exception as e:
                        print(f"Error during focusing: {e}")
                        continue

                    print("Starting image capture...")
                    cmr.save_images(camera, config.camera.num_of_images, frame_dir)
                    print("Image capture complete.")

                except Exception as e:
                    print(f"Error processing circle {idx}: {e}")
                finally:
                    try:
                        camera.OffsetX.Value = 0
                        camera.OffsetY.Value = 0
                        camera.Width.Value = max_width
                        camera.Height.Value = max_height
                        print("Camera settings reset.")
                    except genicam.GenericException as e:
                        print(f"Error resetting camera settings: {e}")

    print("Closing connections...")
    try:
        pidevice.CloseConnection()
    except Exception as e:
        print(f"Error closing motion controller connection: {e}")

    try:
        camera.Close()
    except genicam.GenericException as e:
        print(f"Error closing camera connection: {e}")

    print("Process complete.")


def adjust(value, increment, max_value):
    raw_value = int(max(0, value))
    adjusted_value = raw_value - (raw_value % increment)
    final_value = min(adjusted_value, max_value)
    return final_value


if __name__ == "__main__":
    main()
