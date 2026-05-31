import pandas as pd
import numpy as np
import os

np.random.seed(42)

def generate_cars_data(n=3000):
    """Генерация синтетических данных для датасета cars (регрессия)"""
    brands = ['Toyota', 'Honda', 'BMW', 'Mercedes', 'Audi', 'Volkswagen', 'Ford', 'Hyundai', 'Kia', 'Nissan']
    models_map = {
        'Toyota': ['Corolla', 'Camry', 'RAV4', 'Land Cruiser'],
        'Honda': ['Civic', 'Accord', 'CR-V'],
        'BMW': ['3 Series', '5 Series', 'X3', 'X5'],
        'Mercedes': ['C-Class', 'E-Class', 'GLC', 'GLE'],
        'Audi': ['A4', 'A6', 'Q5', 'Q7'],
        'Volkswagen': ['Golf', 'Passat', 'Tiguan', 'Touareg'],
        'Ford': ['Focus', 'Mondeo', 'Kuga'],
        'Hyundai': ['Solaris', 'Elantra', 'Tucson', 'Santa Fe'],
        'Kia': ['Rio', 'Cerato', 'Sportage', 'Sorento'],
        'Nissan': ['Almera', 'Teana', 'X-Trail', 'Patrol']
    }
    transmissions = ['Manual', 'Automatic', 'CVT', 'Robot']
    body_types = ['Sedan', 'Hatchback', 'SUV', 'Wagon', 'Coupe']
    fuel_types = ['Petrol', 'Diesel', 'Hybrid', 'Electric']
    colors = ['White', 'Black', 'Silver', 'Grey', 'Blue', 'Red', 'Green', 'Brown']

    data = []
    for _ in range(n):
        brand = np.random.choice(brands)
        model = np.random.choice(models_map[brand])
        year = np.random.randint(2000, 2024)
        mileage = np.random.randint(0, 300000)
        engine_capacity = np.random.choice([1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.5, 2.7, 3.0, 3.5, 4.0, 5.0])
        transmission = np.random.choice(transmissions)
        body_type = np.random.choice(body_types)
        fuel_type = np.random.choice(fuel_types)
        color = np.random.choice(colors)
        doors = np.random.choice([2, 3, 4, 5])
        
        # Формирование цены на основе признаков с некоторым шумом
        base_price = 10000
        year_factor = (year - 2000) * 800
        mileage_factor = max(0, (1 - mileage / 200000)) * 15000
        engine_factor = engine_capacity * 3000
        brand_premium = {'Toyota': 2000, 'Honda': 1500, 'BMW': 8000, 'Mercedes': 9000, 
                         'Audi': 7500, 'Volkswagen': 3000, 'Ford': 1500, 'Hyundai': 1000, 
                         'Kia': 1000, 'Nissan': 1500}[brand]
        body_premium = {'Sedan': 1000, 'Hatchback': 0, 'SUV': 4000, 'Wagon': 500, 'Coupe': 2000}[body_type]
        fuel_premium = {'Petrol': 0, 'Diesel': 1500, 'Hybrid': 5000, 'Electric': 8000}[fuel_type]
        
        price_usd = base_price + year_factor + mileage_factor + engine_factor + brand_premium + body_premium + fuel_premium
        price_usd += np.random.normal(0, price_usd * 0.1)  # 10% шум
        price_usd = max(2000, price_usd)
        
        data.append([brand, model, year, mileage, engine_capacity, transmission, body_type, fuel_type, color, doors, round(price_usd, 2)])
    
    df = pd.DataFrame(data, columns=['brand', 'model', 'year', 'mileage', 'engine_capacity', 
                                      'transmission', 'body_type', 'fuel_type', 'color', 'doors', 'price_usd'])
    return df

def generate_diabetes_data(n=5000):
    """Генерация синтетических данных для датасета Diabetes (классификация)"""
    # Diabetes_012: 0=нет диабета, 1=преддиабет, 2=диабет
    data = []
    for _ in range(n):
        high_bp = np.random.choice([0, 1], p=[0.6, 0.4])
        high_chol = np.random.choice([0, 1], p=[0.55, 0.45])
        chol_check = np.random.choice([0, 1], p=[0.2, 0.8])
        bmi = np.random.uniform(15, 55)
        smoker = np.random.choice([0, 1], p=[0.65, 0.35])
        stroke = np.random.choice([0, 1], p=[0.92, 0.08])
        heart_disease = np.random.choice([0, 1], p=[0.88, 0.12])
        phys_activity = np.random.choice([0, 1], p=[0.25, 0.75])
        fruits = np.random.choice([0, 1], p=[0.3, 0.7])
        veggies = np.random.choice([0, 1], p=[0.25, 0.75])
        hvy_alcohol = np.random.choice([0, 1], p=[0.7, 0.3])
        healthcare = np.random.choice([0, 1], p=[0.1, 0.9])
        no_doc_cost = np.random.choice([0, 1], p=[0.85, 0.15])
        gen_hlth = np.random.randint(1, 6)
        ment_hlth = np.random.randint(0, 31)
        phys_hlth = np.random.randint(0, 31)
        diff_walk = np.random.choice([0, 1], p=[0.75, 0.25])
        sex = np.random.choice([0, 1], p=[0.45, 0.55])
        age = np.random.randint(1, 14)  # 1=18-24, 13=80+
        education = np.random.randint(1, 7)
        income = np.random.randint(1, 9)
        
        # Логика классификации
        score = 0
        if high_bp: score += 1
        if high_chol: score += 1
        if bmi > 30: score += 2
        if smoker: score += 1
        if heart_disease: score += 2
        if phys_activity == 0: score += 1
        if gen_hlth >= 4: score += 2
        if diff_walk: score += 1
        if age >= 10: score += 1
        if income <= 3: score += 1
        
        if score <= 2:
            diabetes = 0
        elif score <= 5:
            diabetes = 1
        else:
            diabetes = 2
        
        # Добавляем некоторую случайность
        if np.random.random() < 0.15:
            diabetes = np.random.choice([0, 1, 2])
        
        data.append([high_bp, high_chol, chol_check, round(bmi, 1), smoker, stroke, heart_disease,
                     phys_activity, fruits, veggies, hvy_alcohol, healthcare, no_doc_cost, gen_hlth,
                     ment_hlth, phys_hlth, diff_walk, sex, age, education, income, diabetes])
    
    df = pd.DataFrame(data, columns=['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
                                     'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                                     'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                                     'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
                                     'Income', 'Diabetes_012'])
    return df

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    
    cars_df = generate_cars_data(3000)
    cars_df.to_csv('data/cars.csv', index=False)
    print('Сохранен data/cars.csv')
    
    diabetes_df = generate_diabetes_data(5000)
    diabetes_df.to_csv('data/diabetes.csv', index=False)
    print('Сохранен data/diabetes.csv')
