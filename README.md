# EITINN
Inexact Newton regularization for solving the inverse problem of Electrical Impedance Tomography (EIT).


### 🔧 Installation

This project requires [FEniCS](https://fenicsproject.org/) to run.
The recommended installation method is via [Miniconda](https://docs.conda.io/en/latest/miniconda.html) using the provided `environment.yml` file.

1. **Install Miniconda** if you haven't already. You can download it from [here](https://docs.conda.io/en/latest/miniconda.html).
2. **Create and activate the environment** using the provided configuration file:
   ```sh
   conda env create --file environment.yml
   conda activate eitinn
   ```


### Examples

Reconstructions using the [KIT4](https://arxiv.org/abs/1704.01178) dataset.

![REC23](https://raw.githubusercontent.com/mpauleti/eitinn/main/doc/images/rec_2_3.png)
![REC41](https://raw.githubusercontent.com/mpauleti/eitinn/main/doc/images/rec_4_1.png)
![REC44](https://raw.githubusercontent.com/mpauleti/eitinn/main/doc/images/rec_4_4.png)


### 📖 References

This code was based on the following [book](https://coloquio34.impa.br/pdf/34CBM07-eBook.pdf) (in Portuguese).
The book repository is available [here](https://github.com/HafemannE/FEIT_CBM34).


### 📘 Citation

If you use this program in your work, please cite:

```
@article{pmr2023,
  title={Inexact Newton regularizations with uniformly convex stability terms: a unified convergence analysis},
  author={Pauleti, Marco and Margotti, F{\'a}bio and Rieder, Andreas},
  year={2023},
  url={https://doi.org/10.5445/IR/1000157900}
}
```
