**Image Compression using 2D Walsh Hadamard Transform**

This code implements image compression using the 2D Walsh Hadamard Transform (WHT). The WHT is applied to a grayscale image, and the resulting coefficients are truncated to achieve compression. The compressed image is then reconstructed using the inverse WHT.

**Requirements**

numpy
OpenCV

**Usage**

Enter the filename (with extension) of the grayscale image.
The program will calculate the PSNR of the compressed image and display the results.
The compressed image and original image will be displayed.
Note: The compressed image will be saved as imgcompressed.jpg in the current directory.

**Example**

An example image named example.png is included in the repository. To run the code on this image, simply run the wh_image_compression.py file and enter example.png when prompted for the filename.