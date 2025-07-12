LeadFocal: Emotion & Location Recognition System ================================================= 

Overview: 
--------- 
This is a prototype security interface that simulates emotion and location recognition based on uploaded images. It was developed for the National Security Innovation Challenge 2025. 

Project Team: 
------------- 
Team ID: [Your Team ID] 
Member C: System integration, Streamlit frontend interface, and historical record display 

Main Features: 
-------------- 
- Username-based session identification 
- Image upload and display 
- Simulated emotion detection (e.g., Happy, Sad, Angry) 
- Simulated location estimation (e.g., Kuala Lumpur, Tokyo) 
- Automatic saving of results to a CSV file 
- History page for viewing past uploads and filtering by user 

How to Run Locally: 
------------------- 
1. Install required packages: pip install streamlit pandas 
2. Run the application: streamlit run app.py 
3. The browser will open at: http://localhost:8501/ 

Deployment Notes: 
----------------- 
This app is deployable on Streamlit Cloud. Just upload the code repository (with app.py and optional history.csv) to GitHub and deploy via https://streamlit.io/cloud. 

File Structure: 
--------------- 
- app.py → Main Streamlit application 
- history.csv → Automatically created for storing upload records 
- README.txt → System overview and instructions Thank you for reviewing our project!
