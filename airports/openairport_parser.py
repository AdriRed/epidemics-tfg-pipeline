# airports data file from https://github.com/jpatokal/openflights/blob/master/data/routes.dat

import csv
from collections import defaultdict
import argparse

def procesar_rutas(archivo_entrada, archivo_salida, delimiter=','):
    """
    Procesa un archivo CSV de rutas aéreas y genera un archivo con conexiones no-dirigidas.
    
    Args:
        archivo_entrada (str): Ruta al archivo CSV de entrada
        archivo_salida (str): Ruta al archivo de salida
        delimiter (str): Delimitador del archivo CSV (por defecto ',')
    """
    
    # Diccionario para almacenar las conexiones y sus pesos
    # Usamos una tupla ordenada (id_menor, id_mayor) como clave
    conexiones = defaultdict(int)
    
    try:
        with open(archivo_entrada, 'r', encoding='utf-8') as file:
            # Leer el archivo CSV
            reader = csv.reader(file, delimiter=delimiter)
            
            # Leer la cabecera si existe
            try:
                header = next(reader)
                print(f"Cabecera encontrada: {header}")
            except StopIteration:
                print("El archivo está vacío")
                return
            
            # Contar líneas procesadas
            lineas_procesadas = 0
            lineas_con_error = 0
            
            for fila in reader:
                try:
                    # Verificar que la fila tiene al menos 8 columnas
                    if len(fila) < 8:
                        print(f"Advertencia: Fila con {len(fila)} columnas ignorada: {fila}")
                        lineas_con_error += 1
                        continue
                    
                    # Extraer los IDs de los aeropuertos (columnas 3 y 5 considerando 0-index)
                    # Ajusta estos índices según la estructura real de tu CSV
                    aeropuerto1_id = fila[3].strip()  # Source airport ID
                    aeropuerto2_id = fila[5].strip()  # Destination airport ID
                    if (aeropuerto1_id == '\\N' or aeropuerto2_id == '\\N'):
                        continue
                    # Validar que los IDs no estén vacíos
                    if not aeropuerto1_id or not aeropuerto2_id:
                        print(f"Advertencia: IDs vacíos en fila: {fila}")
                        lineas_con_error += 1
                        continue
                    
                    # Crear clave ordenada para conexión no-dirigida
                    if aeropuerto1_id < aeropuerto2_id:
                        clave = (aeropuerto1_id, aeropuerto2_id)
                    else:
                        clave = (aeropuerto2_id, aeropuerto1_id)
                    
                    # Incrementar el peso de la conexión
                    conexiones[clave] += 1
                    
                    lineas_procesadas += 1
                    
                except Exception as e:
                    print(f"Error procesando fila: {fila}")
                    print(f"Error: {e}")
                    lineas_con_error += 1
            
            print(f"\nProcesadas {lineas_procesadas} rutas")
            print(f"Líneas con error: {lineas_con_error}")
            print(f"Conexiones únicas encontradas: {len(conexiones)}")
    
    except FileNotFoundError:
        print(f"Error: No se encuentra el archivo {archivo_entrada}")
        return
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return
    
    # Escribir el archivo de salida
    try:
        with open(archivo_salida, 'w', encoding='utf-8') as file:
            # Escribir cabecera
            file.write("ID_Aeropuerto1 ID_Aeropuerto2 Peso\n")
            
            # Escribir conexiones ordenadas
            for (aero1, aero2), peso in sorted(conexiones.items()):
                file.write(f"{aero1} {aero2} {peso}\n")
        
        print(f"\nArchivo de salida generado: {archivo_salida}")
        print(f"Total de conexiones escritas: {len(conexiones)}")
        
    except Exception as e:
        print(f"Error al escribir el archivo de salida: {e}")

def procesar_rutas_con_airline_filter(archivo_entrada, archivo_salida, delimiter=','):
    """
    Versión alternativa que permite filtrar por aerolínea específica
    """
    
    conexiones = defaultdict(int)
    aerolineas_rutas = defaultdict(list)  # Para estadísticas adicionales
    
    try:
        with open(archivo_entrada, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=delimiter)
            next(reader)  # Saltar cabecera
            
            for fila in reader:
                if len(fila) >= 8:
                    airline_id = fila[1]  # Airline ID
                    aero1_id = fila[3]
                    aero2_id = fila[5]
                    
                    if aero1_id and aero2_id:
                        # Ordenar IDs para conexión no-dirigida
                        if aero1_id < aero2_id:
                            clave = (aero1_id, aero2_id)
                        else:
                            clave = (aero2_id, aero1_id)
                        
                        conexiones[clave] += 1
                        aerolineas_rutas[clave].append(airline_id)
        
        # Escribir archivo de salida con información adicional
        with open(archivo_salida, 'w', encoding='utf-8') as file:
            file.write("ID_Aeropuerto1 ID_Aeropuerto2 Peso Aerolineas_Distintas\n")
            
            for (aero1, aero2), peso in sorted(conexiones.items()):
                aerolineas_distintas = len(set(aerolineas_rutas[(aero1, aero2)]))
                file.write(f"{aero1} {aero2} {peso} {aerolineas_distintas}\n")
        
        print(f"Archivo generado con estadísticas adicionales: {archivo_salida}")
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Procesar archivo CSV de rutas aéreas')
    parser.add_argument('archivo_entrada', help='Archivo CSV de entrada')
    parser.add_argument('archivo_salida', help='Archivo de salida')
    parser.add_argument('--delimiter', default=',', help='Delimitador del CSV (default: ,)')
    parser.add_argument('--advanced', action='store_true', 
                       help='Usar procesamiento avanzado con estadísticas de aerolíneas')
    
    args = parser.parse_args()
    
    if args.advanced:
        procesar_rutas_con_airline_filter(args.archivo_entrada, args.archivo_salida, args.delimiter)
    else:
        procesar_rutas(args.archivo_entrada, args.archivo_salida, args.delimiter)

if __name__ == "__main__":
    # Si no se proporcionan argumentos, usar valores por defecto para pruebas
    import sys
    if len(sys.argv) == 1:
        print("Uso: python script.py archivo_entrada.csv archivo_salida.txt [--delimiter ,] [--advanced]")
        print("\nEjecutando con valores por defecto:")
        print("python script.py rutas.csv conexiones.txt")
        
        # Aquí puedes poner los nombres de archivo que quieras usar por defecto
        archivo_entrada = "rutas.csv"
        archivo_salida = "conexiones.txt"
        
        if input(f"\n¿Procesar {archivo_entrada} -> {archivo_salida}? (s/n): ").lower() == 's':
            procesar_rutas(archivo_entrada, archivo_salida)
    else:
        main()