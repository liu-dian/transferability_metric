from tool.metric import *

if __name__ == '__main__':
    # Define a list of directory sets
    directory_sets = [
        {
            'tar_root_dir': 'dataset/tar_dslr_task0',
            'tar_predictz_dir': 'dataset/tar_dslr_task0_predictz',
            'src_root_dir': 'dataset/src_Office31_amazon_task0_sampled200'
        },
        {
            'tar_root_dir': 'dataset/tar_webcam_task1',
            'tar_predictz_dir': 'dataset/tar_webcam_task1_predictz',
            'src_root_dir': 'dataset/src_Office31_amazon_task0_sampled200'
        },
        # Add more directory sets here
    ]

    # Initialize a dictionary to hold the results
    results = {}

    # Iterate over each directory set
    for i, dirs in enumerate(directory_sets):
        # Calculate each metric and store the result in the results dictionary
        results[i] = {
            'h_score': h_score(dirs['tar_root_dir']),
            'log_expected_empirical_prediction': log_expected_empirical_prediction(dirs['tar_predictz_dir']),
            'log_maximum_evidence': log_maximum_evidence(dirs['tar_root_dir']),
            'negative_conditional_entropy': negative_conditional_entropy(dirs['src_root_dir'], dirs['tar_predictz_dir']),
            'optimal_transport': optimal_transport(dirs['src_root_dir'], dirs['tar_root_dir'])
        }

    # Print the results
    for i, result in results.items():
        print(f"Results for directory set {i}:")
        for metric, value in result.items():
            print(f"{metric}: {value}")
