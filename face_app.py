import cv2
import face_recognition
import numpy as np
import os
import pickle
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import threading
from datetime import datetime

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.model_file = "face_model.pkl"
        self.training_data_dir = "training_faces"
        self.confidence_threshold = 0.6  # Threshold untuk menentukan kecocokan
        
        # Buat direktori training jika belum ada
        if not os.path.exists(self.training_data_dir):
            os.makedirs(self.training_data_dir)
        
        # Load model yang sudah ada
        self.load_model()
        
        # Inisialisasi variabel GUI
        self.root = None
        self.info_text = None
        self.status_var = None
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Sistem Pengenalan Wajah - Tugas Machine Learning")
        self.root.geometry("800x600")
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Sistem Pengenalan Wajah", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Training section
        training_frame = ttk.LabelFrame(main_frame, text="Pelatihan Model", padding="10")
        training_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(training_frame, text="Pilih Foto untuk Training", 
                  command=self.add_training_face).grid(row=0, column=0, padx=5)
        
        ttk.Button(training_frame, text="Ambil Foto dari Webcam", 
                  command=self.capture_from_webcam).grid(row=0, column=1, padx=5)
        
        ttk.Button(training_frame, text="Latih Model", 
                  command=self.train_model).grid(row=0, column=2, padx=5)
        
        ttk.Button(training_frame, text="Reset Model", 
                  command=self.reset_model).grid(row=0, column=3, padx=5)
        
        # Detection section
        detection_frame = ttk.LabelFrame(main_frame, text="Deteksi Wajah", padding="10")
        detection_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(detection_frame, text="Mulai Webcam", 
                  command=self.start_webcam).grid(row=0, column=0, padx=5)
        
        ttk.Button(detection_frame, text="Unggah Gambar", 
                  command=self.upload_image).grid(row=0, column=1, padx=5)
        
        ttk.Button(detection_frame, text="Hentikan Webcam", 
                  command=self.stop_webcam).grid(row=0, column=2, padx=5)
        
        # Info section
        info_frame = ttk.LabelFrame(main_frame, text="Informasi Model", padding="10")
        info_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.info_text = tk.Text(info_frame, height=8, width=80)
        scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Status
        self.status_var = tk.StringVar()
        self.status_var.set("Siap - Silakan tambahkan foto training terlebih dahulu")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=4, column=0, columnspan=3, pady=10)
        
        # Variables untuk webcam
        self.webcam_running = False
        self.cap = None
        self.capture_webcam_running = False
        self.capture_name = ""
        
        self.update_info()
        
    def log_info(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.info_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.info_text.see(tk.END)
        self.root.update()
        
    def update_info(self):
        info = f"Status Model:\n"
        info += f"- Jumlah wajah yang dikenali: {len(self.known_face_names)}\n"
        info += f"- Nama yang dikenali: {', '.join(set(self.known_face_names)) if self.known_face_names else 'Belum ada'}\n"
        info += f"- Threshold confidence: {self.confidence_threshold}\n"
        info += f"- File model: {'✓ Ada' if os.path.exists(self.model_file) else '✗ Belum ada'}\n"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, info)
        
    def add_training_face(self):
        file_paths = filedialog.askopenfilenames(
            title="Pilih foto untuk training",
            filetypes=[("File Gambar", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not file_paths:
            return
            
        name = simpledialog.askstring("Input", "Masukkan nama pemilik wajah:")
        if not name:
            return
            
        success_count = 0
        for file_path in file_paths:
            if self.process_training_image(file_path, name):
                success_count += 1
                
        if success_count > 0:
            self.log_info(f"Berhasil menambahkan {success_count} foto training untuk '{name}'")
            messagebox.showinfo("Sukses", f"Berhasil menambahkan {success_count} foto training!")
        else:
            messagebox.showerror("Error", "Tidak ada wajah yang terdeteksi di foto yang dipilih!")
            
    def capture_from_webcam(self):
        self.capture_name = simpledialog.askstring("Input", "Masukkan nama pemilik wajah:")
        if not self.capture_name:
            return
            
        self.capture_webcam_running = True
        self.status_var.set("Webcam Capture aktif - Tekan 'C' untuk capture, 'Q' untuk keluar")
        
        def capture_thread():
            cap = cv2.VideoCapture(0)
            capture_count = 0
            
            while self.capture_webcam_running:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Tampilkan instruksi di frame
                cv2.putText(frame, "Tekan 'C' untuk Capture, 'Q' untuk Keluar", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Tampilkan frame
                cv2.imshow('Capture Training Image - Press C to Capture', frame)
                
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                elif key & 0xFF == ord('c'):
                    # Coba deteksi wajah
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame)
                    
                    if len(face_locations) > 0:
                        # Simpan gambar
                        filename = f"{self.capture_name}_{len(self.known_face_names)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        save_path = os.path.join(self.training_data_dir, filename)
                        cv2.imwrite(save_path, frame)
                        
                        capture_count += 1
                        self.log_info(f"Foto training {capture_count} disimpan: {filename}")
                        messagebox.showinfo("Sukses", f"Foto berhasil diambil dan disimpan sebagai {filename}")
                    else:
                        messagebox.showerror("Error", "Tidak ada wajah yang terdeteksi!")
            
            cap.release()
            cv2.destroyAllWindows()
            self.capture_webcam_running = False
            self.status_var.set("Siap")
            
            if capture_count > 0:
                messagebox.showinfo("Info", f"Berhasil mengambil {capture_count} foto untuk training '{self.capture_name}'")
            
        threading.Thread(target=capture_thread, daemon=True).start()
            
    def process_training_image(self, image_path, name):
        try:
            # Load dan encode gambar
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) == 0:
                self.log_info(f"Tidak ada wajah terdeteksi di {os.path.basename(image_path)}")
                return False
                
            # Ambil encoding wajah pertama
            face_encoding = face_encodings[0]
            
            # Simpan ke data training
            filename = f"{name}_{len(self.known_face_names)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            training_path = os.path.join(self.training_data_dir, filename)
            
            # Copy gambar ke folder training
            import shutil
            shutil.copy2(image_path, training_path)
            
            self.log_info(f"Foto training disimpan: {filename}")
            return True
            
        except Exception as e:
            self.log_info(f"Error processing {image_path}: {str(e)}")
            return False
            
    def train_model(self):
        self.log_info("Memulai training model...")
        self.status_var.set("Training model...")
        
        # Reset data
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Load semua gambar dari folder training
        training_files = [f for f in os.listdir(self.training_data_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not training_files:
            messagebox.showerror("Error", "Tidak ada foto training! Tambahkan foto terlebih dahulu.")
            return
            
        success_count = 0
        for filename in training_files:
            file_path = os.path.join(self.training_data_dir, filename)
            
            # Extract nama dari filename
            name = filename.split('_')[0]
            
            try:
                image = face_recognition.load_image_file(file_path)
                face_encodings = face_recognition.face_encodings(image)
                
                if len(face_encodings) > 0:
                    self.known_face_encodings.append(face_encodings[0])
                    self.known_face_names.append(name)
                    success_count += 1
                    self.log_info(f"Training: {filename} -> {name}")
                    
            except Exception as e:
                self.log_info(f"Error training {filename}: {str(e)}")
                
        if success_count > 0:
            # Simpan model
            self.save_model()
            self.log_info(f"Training selesai! {success_count} wajah berhasil ditraining.")
            self.status_var.set(f"Model siap - {len(set(self.known_face_names))} orang dikenali")
            messagebox.showinfo("Sukses", f"Training selesai!\n{success_count} wajah berhasil ditraining.")
        else:
            messagebox.showerror("Error", "Training gagal! Tidak ada wajah yang berhasil diproses.")
            
        self.update_info()
        
    def save_model(self):
        try:
            model_data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names,
                'threshold': self.confidence_threshold
            }
            
            with open(self.model_file, 'wb') as f:
                pickle.dump(model_data, f)
                
            self.log_info(f"Model disimpan ke {self.model_file}")
            
        except Exception as e:
            self.log_info(f"Error menyimpan model: {str(e)}")
            
    def load_model(self):
        try:
            if os.path.exists(self.model_file):
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.known_face_encodings = model_data['encodings']
                self.known_face_names = model_data['names']
                self.confidence_threshold = model_data.get('threshold', 0.6)
                
                self.log_info(f"Model berhasil dimuat dari {self.model_file}")
                self.log_info(f"Jumlah wajah yang dikenali: {len(self.known_face_names)}")
                
        except Exception as e:
            self.log_info(f"Error memuat model: {str(e)}")
            
    def reset_model(self):
        if messagebox.askyesno("Konfirmasi", "Yakin ingin reset model dan hapus semua data training?"):
            self.known_face_encodings = []
            self.known_face_names = []
            
            # Hapus file model
            if os.path.exists(self.model_file):
                os.remove(self.model_file)
                
            # Hapus folder training
            if os.path.exists(self.training_data_dir):
                import shutil
                shutil.rmtree(self.training_data_dir)
                os.makedirs(self.training_data_dir)
                
            self.log_info("Model dan data training berhasil direset")
            self.status_var.set("Model direset - Silakan tambahkan foto training")
            self.update_info()
            
    def recognize_faces(self, image):
        # Resize untuk performa lebih baik
        small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        rgb_small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)
        
        # Deteksi wajah
        face_locations = face_recognition.face_locations(rgb_small_image)
        face_encodings = face_recognition.face_encodings(rgb_small_image, face_locations)
        
        results = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale kembali lokasi wajah
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            name = "Tidak Dikenali"
            confidence = 0
            
            if len(self.known_face_encodings) > 0:
                # Hitung jarak dengan semua wajah yang dikenal
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                # Konversi jarak ke confidence (0-100%)
                distance = face_distances[best_match_index]
                confidence = max(0, (1 - distance) * 100)
                
                # Tentukan nama berdasarkan threshold
                if distance <= self.confidence_threshold:
                    name = self.known_face_names[best_match_index]
                    
            results.append({
                'name': name,
                'confidence': confidence,
                'location': (top, right, bottom, left)
            })
            
        return results
        
    def start_webcam(self):
        if not self.known_face_encodings:
            messagebox.showerror("Error", "Model belum ditraining! Tambahkan foto training terlebih dahulu.")
            return
            
        self.webcam_running = True
        self.status_var.set("Webcam aktif - Deteksi wajah berjalan")
        
        def webcam_thread():
            self.cap = cv2.VideoCapture(0)
            
            while self.webcam_running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                # Deteksi wajah
                results = self.recognize_faces(frame)
                
                # Gambar hasil deteksi
                for result in results:
                    top, right, bottom, left = result['location']
                    name = result['name']
                    confidence = result['confidence']
                    
                    # Warna box berdasarkan status
                    color = (0, 255, 0) if name != "Tidak Dikenali" else (0, 0, 255)
                    
                    # Gambar rectangle
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    # Label dengan nama dan confidence
                    label = f"{name} ({confidence:.1f}%)"
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                    cv2.putText(frame, label, (left + 6, bottom - 6), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                
                # Tampilkan frame
                cv2.imshow('Pengenalan Wajah - Tekan Q untuk keluar', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            self.cap.release()
            cv2.destroyAllWindows()
            
        threading.Thread(target=webcam_thread, daemon=True).start()
        
    def stop_webcam(self):
        self.webcam_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.status_var.set("Webcam dihentikan")
        
    def upload_image(self):
        if not self.known_face_encodings:
            messagebox.showerror("Error", "Model belum ditraining! Tambahkan foto training terlebih dahulu.")
            return
            
        file_path = filedialog.askopenfilename(
            title="Pilih gambar untuk dideteksi",
            filetypes=[("File Gambar", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not file_path:
            return
            
        try:
            # Load gambar
            image = cv2.imread(file_path)
            
            # Deteksi wajah
            results = self.recognize_faces(image)
            
            if not results:
                messagebox.showinfo("Hasil", "Tidak ada wajah yang terdeteksi dalam gambar!")
                return
                
            # Gambar hasil deteksi
            for result in results:
                top, right, bottom, left = result['location']
                name = result['name']
                confidence = result['confidence']
                
                color = (0, 255, 0) if name != "Tidak Dikenali" else (0, 0, 255)
                
                cv2.rectangle(image, (left, top), (right, bottom), color, 2)
                
                label = f"{name} ({confidence:.1f}%)"
                cv2.rectangle(image, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(image, label, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # Resize untuk tampilan jika terlalu besar
            height, width = image.shape[:2]
            if width > 800:
                scale = 800 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            # Tampilkan hasil
            cv2.imshow('Hasil Deteksi Wajah - Tekan sembarang tombol untuk menutup', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Log hasil
            result_text = "Hasil deteksi:\n"
            for i, result in enumerate(results, 1):
                result_text += f"Wajah {i}: {result['name']} (Tingkat Kecocokan: {result['confidence']:.1f}%)\n"
            
            self.log_info(result_text)
            messagebox.showinfo("Hasil Deteksi", result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error memproses gambar: {str(e)}")
            
    def run(self):
        self.root.mainloop()

def main():
    print("=== SISTEM PENGENALAN WAJAH ===")
    print("Tugas Machine Learning - Deteksi Wajah")
    print("\nPastikan packages berikut sudah terinstall:")
    print("- opencv-python")
    print("- face-recognition")
    print("- numpy")
    print("- pillow")
    print("- dlib")
    
    print("\nUntuk install packages:")
    print("pip install opencv-python face-recognition numpy pillow dlib")
    print("\nNote: face-recognition membutuhkan dlib yang mungkin perlu CMake")
    print("Jika error, install: pip install cmake")
    
    print("\n" + "="*50)
    
    try:
        app = FaceRecognitionSystem()
        app.run()
    except ImportError as e:
        print(f"Error: {e}")
        print("Silakan install packages yang diperlukan terlebih dahulu!")
    except Exception as e:
        print(f"Error menjalankan aplikasi: {e}")

if __name__ == "__main__":
    main()