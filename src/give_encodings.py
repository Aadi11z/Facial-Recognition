import os

def generate_face_recognition_script(students_list_path, output_file_path, image_directory):
    with open(students_list_path, 'r') as file:
        student_lines = file.readlines()

    # Prepare the output Python file
    with open(output_file_path, 'w') as output_file:
        output_file.write("import face_recognition\n\n")
        
        for line in student_lines:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
                
            try:
                # Clean up the line and split the student ID and name
                student_id, student_name = line.split(" : ")
                student_id = student_id.strip()
                student_name = student_name.strip()
                
                if student_id.startswith('*'):
                    continue
                    
                # Generate variable names for the student based on their ID
                id_part = student_id[8:12]  # Get the last 4 digits
                
                # Image and encoding loading code
                image_var = f"std{id_part}_image"
                encoding_var = f"std{id_part}_encoding"
                
                # Path to the image based on student ID
                image_path = os.path.join(image_directory, f"{student_id}.jpg")
                
                # Write to the output file
                output_file.write(f"{image_var} = face_recognition.load_image_file('{image_path}')\n")
                output_file.write(f"{encoding_var} = face_recognition.face_encodings({image_var})[0]\n\n")
                
            except ValueError as e:
                print(f"Skipping invalid line: {line}")
                continue

    print(f"Python script generated and saved to {output_file_path}")

# Path configurations
students_list_path = ""
image_directory = ""
output_file_path = ""

# Generate the Python script
generate_face_recognition_script(students_list_path, output_file_path, image_directory)