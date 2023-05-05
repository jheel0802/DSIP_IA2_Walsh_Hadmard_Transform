import numpy as np
import cv2

# Load image
filename = input('Enter the filename (with extension) of the grayscale image: ')
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# Get image size
r, c = img.shape

# Convert to double
imgg = np.float64(img)

# Forward WHT
yc = np.apply_along_axis(np.fft.fft, 0, imgg)
yr = np.apply_along_axis(np.fft.fft, 1, yc)
y = yr

# Backup coefficients
yo = y.copy()

# Truncate coefficients
y[256:r, 256:c] = 0

# Inverse WHT
Ir1 = np.apply_along_axis(np.fft.ifft, 0, y)
Ir2 = np.apply_along_axis(np.fft.ifft, 1, Ir1)
imgr = Ir2.real

# Convert to uint8
imgr8 = np.uint8(imgr)

# Save compressed image
cv2.imwrite('imgcompressed.jpg', imgr8)

# Calculate PSNR
mse = np.sum(np.square(imgr - imgg)) / (r * c)
maxp = max(np.max(imgr), np.max(imgg))
psnr = 10 * np.log10(maxp ** 2 / mse)

print(f'PSNR = {psnr}')

# Display results
cv2.imshow('Original Image', img)
cv2.imshow('Compressed Image', imgr8)

cv2.waitKey(0)
cv2.destroyAllWindows()