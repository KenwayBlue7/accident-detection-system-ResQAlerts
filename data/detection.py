class Detection:
    def __init__(self, box, class_id, confidence=1.0):
        """
        Parameters:
            box (tuple): (x, y, width, height)
            class_id (int): Class index of the detected object
            confidence (float): Detection confidence score
        """
        self.box = box
        self.class_id = class_id
        self.confidence = confidence

    def get_center(self):
        x, y, w, h = self.box
        cx = x + w / 2
        cy = y + h / 2
        return (cx, cy)
