import sys
import importlib
import pkg_resources
import yaml

# Mapping from package names to importable module names
package_import_mapping = {
    'python': 'sys',
    'pytorch': 'torch',
    'torchvision': 'torchvision',
    'pytorch-cuda': 'torch',
    # Add more mappings if necessary
}

def get_package_version(package_name):
    """Retrieve the installed version of a package."""
    try:
        if package_name == 'python':
            # Get the Python version
            return '.'.join(map(str, sys.version_info[:3]))
        elif package_name == 'pytorch-cuda':
            # Get CUDA version from torch
            import torch
            return torch.version.cuda
        else:
            # Map the package name to its importable module name
            import_name = package_import_mapping.get(package_name, package_name)
            try:
                # Try importing the module and getting __version__
                module = importlib.import_module(import_name)
                version = getattr(module, '__version__', None)
                if version:
                    return version
            except ImportError:
                pass

            # Try getting the version from pkg_resources
            try:
                return pkg_resources.get_distribution(package_name).version
            except pkg_resources.DistributionNotFound:
                # Try with underscores instead of hyphens
                package_name_alt = package_name.replace('-', '_')
                return pkg_resources.get_distribution(package_name_alt).version
    except Exception:
        return None

def update_versions(environment):
    """Update the package versions in the environment based on installed versions."""
    updated_deps = []
    deps_iter = iter(environment['dependencies'])
    for dep in deps_iter:
        if isinstance(dep, str):
            if dep == 'pip':
                # Keep 'pip' as is and process the next item
                updated_deps.append(dep)
                pip_deps_dict = next(deps_iter, {})
                if isinstance(pip_deps_dict, dict) and 'pip' in pip_deps_dict:
                    updated_pip_deps = []
                    for pip_dep in pip_deps_dict['pip']:
                        if '==' in pip_dep:
                            package, _ = pip_dep.split('==', 1)
                        else:
                            package = pip_dep
                        new_version = get_package_version(package)
                        if new_version:
                            updated_pip_deps.append(f"{package}=={new_version}")
                        else:
                            updated_pip_deps.append(pip_dep)
                    updated_deps.append({'pip': updated_pip_deps})
                else:
                    updated_deps.append(pip_deps_dict)
            else:
                # Regular dependency
                if '=' in dep:
                    package, _ = dep.split('=', 1)
                else:
                    package = dep
                new_version = get_package_version(package)
                if new_version:
                    updated_deps.append(f"{package}={new_version}")
                else:
                    updated_deps.append(dep)
        elif isinstance(dep, dict):
            # Handle dictionaries (e.g., nested pip dependencies)
            updated_deps.append(dep)
        else:
            # Unrecognized format, keep as is
            updated_deps.append(dep)
    return updated_deps

def main():
    # Your original YAML content as a dictionary
    environment = {
        'name': 'foleycrafter',
        'channels': [
            'pytorch',
            'nvidia'
        ],
        'dependencies': [
            'python=3.10',
            'pytorch=2.2.0',
            'torchvision=0.17.0',
            'pytorch-cuda=11.8',
            'pip',
            {'pip': [
                'diffusers==0.25.1',
                'transformers==4.30.2',
                'xformers',
                'imageio==2.33.1',
                'decord==0.6.0',
                'einops',
                'omegaconf',
                'safetensors',
                'gradio',
                'tqdm==4.66.1',
                'soundfile==0.12.1',
                'wandb',
                'moviepy==1.0.3',
                'kornia==0.7.1',
                'h5py==3.7.0'
            ]}
        ]
    }

    # Update the dependencies with actual installed versions
    environment['dependencies'] = update_versions(environment)

    # Output the updated YAML with identical formatting
    print(yaml.dump(environment, default_flow_style=False, sort_keys=False, allow_unicode=True))

if __name__ == '__main__':
    main()
