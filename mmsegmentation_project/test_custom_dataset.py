# simple_check.py
import os
import numpy as np

data_root = '/workspace/GitHub/ratitas/mmsegmentation_project/datasets/'

def check_data_issues():
    """Verifica problemas comunes con los datos"""
    print("🔍 BUSCANDO PROBLEMAS COMUNES")
    
    issues_found = False
    
    for split in ['train', 'valid', 'test']:
        print(f"\n=== {split.upper()} ===")
        
        img_dir = os.path.join(data_root, 'images', split)
        mask_dir = os.path.join(data_root, 'masks', split)
        
        if not os.path.exists(img_dir):
            print(f"❌ No existe directorio de imágenes: {img_dir}")
            issues_found = True
            continue
            
        if not os.path.exists(mask_dir):
            print(f"❌ No existe directorio de máscaras: {mask_dir}") 
            issues_found = True
            continue
        
        # Verificar archivos vacíos
        for img_file in os.listdir(img_dir)[:2]:  # Solo verificar primeros 2
            img_path = os.path.join(img_dir, img_file)
            if os.path.getsize(img_path) == 0:
                print(f"❌ Imagen vacía: {img_file}")
                issues_found = True
        
        # Verificar que hay archivos
        images = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
        masks = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
        
        print(f"Imágenes: {len(images)}, Máscaras: {len(masks)}")
        
        if len(images) == 0:
            print("❌ No hay imágenes en el directorio")
            issues_found = True
            
        if len(masks) == 0:
            print("❌ No hay máscaras en el directorio")
            issues_found = True
    
    return issues_found

def check_file_permissions():
    """Verifica permisos de archivos"""
    print(f"\n🔐 VERIFICANDO PERMISOS")
    
    test_file = os.path.join(data_root, 'images/train')
    if os.path.exists(test_file):
        readable = os.access(test_file, os.R_OK)
        print(f"Directorio readable: {readable}")
        
        # Verificar un archivo específico
        files = os.listdir(test_file)
        if files:
            test_img = os.path.join(test_file, files[0])
            if os.path.exists(test_img):
                print(f"Archivo readable: {os.access(test_img, os.R_OK)}")

if __name__ == "__main__":
    print("🛠️ DIAGNÓSTICO DEL DATASET")
    print("=" * 50)
    
    issues = check_data_issues()
    check_file_permissions()
    
    if not issues:
        print(f"\n🎉 No se encontraron problemas obvios")
        print("💡 El problema puede estar en la configuración de MMSegmentation")
    else:
        print(f"\n⚠️ Se encontraron problemas que deben solucionarse")