import os
from pypdf import PdfReader # или от pypdf import PdfReader

# Укажите точный путь к вашей папке с PDF (DATA_PATH)
data_path = r"C:\projects\MedCHATbot\data" 

print("Начало проверки PDF файлов...")
for filename in os.listdir(data_path):
    if filename.endswith('.pdf'):
        file_path = os.path.join(data_path, filename)
        try:
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                # Пробуем прочитать хотя бы одну страницу
                if len(reader.pages) > 0:
                    _ = reader.pages[0].extract_text()
        except Exception as e:
            print(f"\n❌ НАЙДЕН ПОВРЕЖДЕННЫЙ ФАЙЛ: {filename}")
            print(f"Ошибка: {e}")

print("\nПроверка завершена!")
