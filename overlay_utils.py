import numpy as np
import cv2

def overlay_results(frame_rgb, dets, segs, det_model=None):
    """
    Combines segmentation mask and detection boxes on one frame.
    Input: RGB image
    Output: RGB image with overlays
    """
    overlay = frame_rgb.copy()

    # --- Segmentation overlay ---
    if hasattr(segs[0], "masks") and segs[0].masks is not None:
        for mask in segs[0].masks.data:
            m = mask.cpu().numpy()

            m = cv2.resize(m, (frame_rgb.shape[1], frame_rgb.shape[0]))

            m_bin = (m > 0.5).astype(np.uint8)
            color = np.array([0, 0, 255], dtype=np.uint8)  # blue mask
            colored_mask = np.zeros_like(overlay)
            colored_mask[m_bin == 1] = color
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.6, 0)

    # --- Detection boxes ---
    for det in dets:
        for box in det.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            cls = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())

            if det_model is not None and hasattr(det_model, "names"):
                label = f"{det_model.names[cls]} {conf:.2f}"
            else:
                label = f"Obj {cls} {conf:.2f}"

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(overlay, label, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return overlay
