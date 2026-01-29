#!/usr/bin/env python3
"""Validate the ML project template structure."""

import sys
from pathlib import Path


def check_file_exists(path: Path, description: str) -> bool:
    """Check if a file exists."""
    exists = path.exists()
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists


def main():
    """Main validation function."""
    project_root = Path(__file__).parent
    all_checks = []

    print("=" * 60)
    print("ML Project Template Validation")
    print("=" * 60)

    # Check configuration files
    print("\n📄 Configuration Files:")
    all_checks.append(check_file_exists(project_root / "pyproject.toml", "pyproject.toml"))
    all_checks.append(check_file_exists(project_root / ".gitignore", ".gitignore"))
    all_checks.append(check_file_exists(project_root / "README.md", "README.md"))
    all_checks.append(check_file_exists(project_root / "SETUP.md", "SETUP.md"))
    all_checks.append(check_file_exists(project_root / "Makefile", "Makefile"))
    all_checks.append(check_file_exists(project_root / ".env.example", ".env.example"))

    # Check VS Code configuration
    print("\n💻 VS Code Configuration:")
    all_checks.append(
        check_file_exists(project_root / ".vscode/settings.json", "settings.json")
    )
    all_checks.append(
        check_file_exists(project_root / ".vscode/launch.json", "launch.json")
    )
    all_checks.append(
        check_file_exists(project_root / ".vscode/extensions.json", "extensions.json")
    )

    # Check source structure
    print("\n📦 Source Code Structure:")
    all_checks.append(check_file_exists(project_root / "src/__init__.py", "src/__init__.py"))
    all_checks.append(
        check_file_exists(project_root / "src/models/__init__.py", "src/models/__init__.py")
    )
    all_checks.append(
        check_file_exists(project_root / "src/models/model.py", "src/models/model.py")
    )
    all_checks.append(
        check_file_exists(project_root / "src/data/__init__.py", "src/data/__init__.py")
    )
    all_checks.append(
        check_file_exists(project_root / "src/data/dataset.py", "src/data/dataset.py")
    )
    all_checks.append(
        check_file_exists(project_root / "src/utils/__init__.py", "src/utils/__init__.py")
    )
    all_checks.append(
        check_file_exists(project_root / "src/utils/helpers.py", "src/utils/helpers.py")
    )
    all_checks.append(check_file_exists(project_root / "src/train.py", "src/train.py"))

    # Check notebooks
    print("\n📓 Jupyter Notebooks:")
    all_checks.append(
        check_file_exists(
            project_root / "notebooks/00_quickstart.ipynb", "00_quickstart.ipynb"
        )
    )
    all_checks.append(
        check_file_exists(
            project_root / "notebooks/01_experiment.ipynb", "01_experiment.ipynb"
        )
    )

    # Check tests
    print("\n🧪 Tests:")
    all_checks.append(check_file_exists(project_root / "tests/__init__.py", "tests/__init__.py"))
    all_checks.append(check_file_exists(project_root / "tests/conftest.py", "tests/conftest.py"))
    all_checks.append(
        check_file_exists(project_root / "tests/test_models.py", "tests/test_models.py")
    )
    all_checks.append(check_file_exists(project_root / "tests/test_data.py", "tests/test_data.py"))
    all_checks.append(
        check_file_exists(project_root / "tests/test_utils.py", "tests/test_utils.py")
    )

    # Check directories
    print("\n📁 Data Directories:")
    all_checks.append(check_file_exists(project_root / "data/raw/.gitkeep", "data/raw"))
    all_checks.append(check_file_exists(project_root / "data/processed/.gitkeep", "data/processed"))
    all_checks.append(check_file_exists(project_root / "data/external/.gitkeep", "data/external"))
    all_checks.append(check_file_exists(project_root / "models/.gitkeep", "models"))

    # Check pyproject.toml content
    print("\n⚙️  Configuration Validation:")
    try:
        import tomllib

        with open(project_root / "pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        # Check for key dependencies
        deps = config.get("project", {}).get("dependencies", [])
        required_deps = ["torch", "jupyter", "numpy", "pandas"]
        for dep in required_deps:
            has_dep = any(dep in d for d in deps)
            status = "✓" if has_dep else "✗"
            print(f"{status} Dependency: {dep}")
            all_checks.append(has_dep)

        # Check dev dependencies
        dev_deps = config.get("project", {}).get("optional-dependencies", {}).get("dev", [])
        required_dev = ["ruff", "pytest"]
        for dep in required_dev:
            has_dep = any(dep in d for d in dev_deps)
            status = "✓" if has_dep else "✗"
            print(f"{status} Dev Dependency: {dep}")
            all_checks.append(has_dep)

    except FileNotFoundError:
        print("✗ Error: pyproject.toml not found")
        all_checks.append(False)
    except Exception as e:
        print(f"✗ Error reading pyproject.toml: {type(e).__name__}: {e}")
        all_checks.append(False)

    # Summary
    print("\n" + "=" * 60)
    passed = sum(all_checks)
    total = len(all_checks)
    percentage = (passed / total * 100) if total > 0 else 0

    print(f"Results: {passed}/{total} checks passed ({percentage:.1f}%)")

    if passed == total:
        print("✅ All checks passed! Template is ready to use.")
        return 0
    else:
        print("⚠️  Some checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
