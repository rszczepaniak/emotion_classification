import os

from configuration import UNPACKED_DATA_DIR


def get_images_data(annotations_data_file_name):
    with open(annotations_data_file_name, 'r') as fh:
        lines = fh.readlines()
    i = 0
    parsed = []
    while i < len(lines):
        image_path = os.path.join(UNPACKED_DATA_DIR, lines[i].strip())
        i += 1
        number_of_faces = int(lines[i])
        i += 1
        faces = []
        for _ in range(number_of_faces):
            face_data = list(map(lambda x: float(x), lines[i].split()))
            faces.append({
                'major_axis_radius': face_data[0],
                'minor_axis_radius': face_data[1],
                'angle': face_data[2],
                'center_x': face_data[3],
                'center_y': face_data[4],
                'label': int(face_data[5])
            })
            i += 1
        parsed.append({
            "path_to_image": image_path + ".jpg",
            "numer_of_faces": number_of_faces,
            "faces": faces
        })
    return parsed
