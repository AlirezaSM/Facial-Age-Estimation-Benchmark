import subprocess
import shutil
import os

data_root = '/media/vision/FastStorage-1/alireza-sm/Facial-Age-Estimation-Benchmark/facebase/data'
results_root = '/media/vision/FastStorage-1/alireza-sm/Facial-Age-Estimation-Benchmark/facebase/results'
base_name = 'DRUnder_256x256_resnet50_imagenet_noisy_dldl_v2'
reference_name = 'DRUnder_256x256_resnet50_imagenet_noisy_dldl_v2_reference'
config_path = 'facebase/configs/other/DRUnder_256x256_resnet50_imagenet_noisy_dldl_v2.yaml'

base_command = "python noise_correction/sync_correction.py --config noise_correction/DRUnder.yaml"

def update_alpha_beta(yaml_file_path, alpha, beta):
    # Read the YAML file as plain text
    with open(yaml_file_path, 'r') as file:
        lines = file.readlines()

    # Flag to track if the num_epochs key is found
    found_alpha = False
    found_beta = False

    # Iterate through the lines to find and update num_epochs
    for i, line in enumerate(lines):
        if found_alpha and found_beta:
            break

        if ('alpha:' in line) and (not found_alpha):
            # Update the line with the new epoch value
            lines[i] = f"  alpha: {alpha}\n"
            found_alpha = True

        if ('beta:' in line) and (not found_beta):
            lines[i] = f"  beta: {beta}\n"
            found_beta = True

    # Raise an error if num_epochs is not found
    if (not found_alpha) and (not found_beta):
        raise KeyError("The 'alpha' and 'beta' keys does not exist in the YAML file.")

    # Write the updated lines back to the YAML file
    with open(yaml_file_path, 'w') as file:
        file.writelines(lines)

    print(f"Updated alpha and beta to {alpha} and {beta} in {yaml_file_path}")

alphas = [0.2, 0.5, 0.8]
betas = [0.2, 0.5, 0.8]

# alphas = [0.1]
# betas = [0.9]


# Loop through each combination of config path and seed
for alpha in alphas:
    for beta in betas:
        print('Update config...')
        update_alpha_beta(config_path, alpha, beta)
        print('Copy data...')
        shutil.copytree(os.path.join(data_root, reference_name), os.path.join(data_root, base_name))

        # Format the command
        command = base_command
        print(f"Running command: {command}")
        
        # Execute the command and stream output to the console
        process = subprocess.run(command, shell=True)
        
        # Check if the command was successful
        if process.returncode != 0:
            print(f"Command failed with return code {process.returncode}: {command}")
            break

        shutil.move(os.path.join(data_root, base_name), os.path.join(data_root, base_name+f'_a{int(alpha*10)}_b{int(beta*10)}'))
        shutil.move(os.path.join(results_root, base_name), os.path.join(results_root, base_name+f'_a{int(alpha*10)}_b{int(beta*10)}'))