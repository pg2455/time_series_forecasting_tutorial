### **Setting Up a Conda Environment with Jupyter (Preferred Method)**

1. **Create and activate a Conda environment** for forecasting:

   ```bash
   conda create --name forecasting python=3.10
   conda activate forecasting
   ```

2. **Install Jupyter and IPython kernel** in the environment:

   ```bash
   conda install jupyter
   conda install -c anaconda ipykernel
   ```

3. **Register the environment's kernel** with Jupyter:

   ```bash
   python -m ipykernel install --user --name=forecasting
   ```

4. **Open Jupyter Notebook** and select the `forecasting` kernel to start executing your code.

---

### **Using `venv` (Alternative Method)**

1. **Create and activate a virtual environment** using Python 3.10 or higher:

   ```bash
   python -m venv forecasting
   source forecasting/bin/activate  # On macOS/Linux
   ```

   On Windows:

   ```bash
   forecasting\Scripts\activate
   ```

2. **How to access the virtual environment in Jupyter**:

   If you are using **VSCode**, it might prompt you to install the kernel, in which case the following steps are not necessary. Otherwise, follow these steps to set it up:

   - **Install Jupyter**:

     ```bash
     pip install jupyter
     ```

   - **Register the environment's kernel** with Jupyter:

     ```bash
     python -m ipykernel install --user --name=forecasting
     ```

---

### **Install Required Libraries**

In both the Conda and `venv` setups, you can install other required libraries from `requirements.txt`:

```bash
pip install -r requirements.txt
```