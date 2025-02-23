import os

import torch
from pipython import pitools
from pypylon import genicam

import lib.cmr as cmr
import lib.cnf as cnf
import lib.fcs as fcs
import lib.mtr as mtr
from lib.circle_detection import get_bounding_boxes


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
    camera = cmr.connect_camera(500, config.camera.exposure)
    os.makedirs(config.file.save_dir, exist_ok=True)

    print("Loading the yolov7 model...")
    model = torch.hub.load(
        "WongKinYiu/yolov7",
        "custom",
        config.file.model_path,
        force_reload=True,
        trust_repo=True,
    )

    org_width = camera.Width.Value
    org_height = camera.Height.Value
    print(f"Original Camera Resolution: {org_width}x{org_height}")

    try:
        offset_x_increment = camera.OffsetX.GetInc()
        offset_y_increment = camera.OffsetY.GetInc()
        width_increment = camera.Width.GetInc()
        height_increment = camera.Height.GetInc()

        print(
            f"Width Increment: {width_increment}, Height Increment: {height_increment}"
        )
        print(
            f"OffsetX Increment: {offset_x_increment}, OffsetY Increment: {offset_y_increment}"
        )
    except genicam.GenericException as e:
        print(f"Error retrieving increment values: {e}")
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

            print("Capturing original image...")
            temp_dir = os.path.join(config.file.save_dir, "temp")
            try:
                cmr.save_images(camera, 1, temp_dir)
            except Exception as e:
                print(f"Error capturing image: {e}")
                continue

            print("Detecting circles...")
            temp_file = os.path.join(temp_dir, "1.tiff")
            boxes = get_bounding_boxes(model, temp_file)

            print(f"Detected {len(boxes)} circles.")

            for idx, box in enumerate(boxes):
                frame_dir = os.path.join(
                    config.file.save_dir, f"position({target_x},{target_y})_cell{idx}"
                )

                try:
                    x_min, y_min, x_max, y_max = box[:4]
                    print(
                        f"\nProcessing Circle {idx}: top left corner ({x_min}, {y_min}), bottom right corner({x_max}, {y_max})"
                    )

                    # NOTE: if you want a specific size uncomment this and comment the other paragraph
                    # Adjusting camera region of interest
                    # camera.Width.Value = config.camera.kernel_size[0]
                    # camera.Height.Value = config.camera.kernel_size[1]

                    camera.Width.Value = x_max - x_min
                    camera.Height.Value = y_max - y_min

                    raw_offset_x = int(max(0, x_min))
                    raw_offset_y = int(max(0, y_min))

                    adjusted_offset_x = adjust_offset(raw_offset_x, offset_x_increment)
                    adjusted_offset_y = adjust_offset(raw_offset_y, offset_y_increment)

                    max_offset_x = camera.OffsetX.GetMax()
                    max_offset_y = camera.OffsetY.GetMax()

                    adjusted_offset_x = min(adjusted_offset_x, max_offset_x)
                    adjusted_offset_y = min(adjusted_offset_y, max_offset_y)

                    camera.OffsetX.Value = adjusted_offset_x
                    camera.OffsetY.Value = adjusted_offset_y

                    print(
                        f"Adjusted Camera ROI: Width={camera.Width.Value}, Height={camera.Height.Value}, "
                        f"OffsetX={camera.OffsetX.Value}, OffsetY={camera.OffsetY.Value}"
                    )

                    print("Performing autofocus...")
                    fcs.move_to_focus(
                        pidevice, camera, config, (x_min, y_min, x_max, y_max)
                    )

                    print("Starting image capture...")
                    cmr.save_images(camera, 10, frame_dir)
                    print("Image capture complete.")

                except Exception as e:
                    print(f"Error processing circle {idx}: {e}")
                finally:
                    try:
                        camera.OffsetX.Value = 0
                        camera.OffsetY.Value = 0
                        camera.Width.Value = org_width
                        camera.Height.Value = org_height
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


def adjust_offset(value, increment):
    return value - (value % increment)


if __name__ == "__main__":
    main()
