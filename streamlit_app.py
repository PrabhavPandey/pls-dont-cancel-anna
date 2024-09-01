import streamlit as st
import subprocess
import os
import time

def run_script(script_name):
    start_time = time.time()
    result = subprocess.run(['python', f'scripts/{script_name}'], capture_output=True, text=True)
    end_time = time.time()
    return result.stdout, end_time - start_time

def main():
    st.title('Namma Yatri - Driver Cancellation and Fare Estimator')
    
    # Check if setup has been run
    if not os.path.exists('setup_complete.txt'):
        st.info("Running initial setup. This may take a few minutes...")
        
        # Run EDA script
        eda_output, eda_time = run_script('eda.py')
        st.success(f"EDA completed in {eda_time:.2f} seconds")
        
        # Run model training script
        model_output, model_time = run_script('model_training.py')
        st.success(f"Model training completed in {model_time:.2f} seconds")
        
        # Create a file to indicate setup is complete
        with open('setup_complete.txt', 'w') as f:
            f.write('Setup completed successfully')
        
        st.success("Setup completed successfully! Launching the main application...")
        time.sleep(3)  # Give user time to read the message
        st.experimental_rerun()
    else:
        # Run the main app
        exec(open("app.py").read())
        
if __name__ == "__main__":
    main()