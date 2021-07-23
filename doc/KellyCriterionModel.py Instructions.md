1. Run file in python console
2. Instantiate Class with variables and provide a path to the Excel file you will be either creating or appending to (can change code to set defaults):
- Ensure Excel file which will be modified is closed in Excel (and any other applications)
- $ KM =   KellyCriterion(ExcelFilePath, variable=1234, variable2=456)
3. Run simulations with: $ KM.multi_sim_to_excel(Number_of_simulations, ExcelSheetName)
4. Change variables in case you want to run simulations with new variables: $ KM.variable=4321
5. Quit code before opening Excel File
6. Histograms and Data available in Excel: 
![image](https://user-images.githubusercontent.com/61990860/126847466-cfe17132-1ce1-4f18-97af-2d1b91bcf09d.png)
![image](https://user-images.githubusercontent.com/61990860/126847500-3e70598e-40ba-4fd8-8580-6328c54a3f2c.png)

