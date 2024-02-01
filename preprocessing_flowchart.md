```

# MRI Image Processing Script Data Flow

[ Main Function: main() ]
    |-> Parse Input Arguments
    |   [ Input: NIfTI file or folder (--input) ]
    |   [ Use T2 or T1 template (--t2, --t1) ]
    |   [ Output folder (--output) ]
    |
    |-> Check and Download Model If Needed
    |   [ Function: download_model_if_needed(templates_folder) ]
    |
    |-> Determine Fixed Image Path
    |   [ Select T1 or T2 template based on arguments ]
    |
    v
[ Instantiate MRIProcessor Class ]
    |-> Input: Parsed arguments (input, output, fixed_image_path)
    |
    |-> Process Files
    |   [ Method: process_files() ]
    |   [ Process each file using affine registration ]
    |
    v
[ Affine Registration for Each File ]
    |-> Method: affine_registration(moving_image_path, output_path)
    |   [ Read fixed and moving images ]
    |   [ Perform affine registration using AntsPy ]
    |   [ Save the registered image to output path ]
    |
    v
[ Completion ]
    |-> Output: "MRI preprocessing is complete."

```