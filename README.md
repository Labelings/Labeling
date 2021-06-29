# Labeling in Python

This package is intended to provide an easy way to store Labelings for image segmentation.
The segmentation information is stored in a BSON file for fast I/O on demand, but can also be exported to JSON for human readability.

### Usage
1. Create a Labeling object with one of the static methods or just init normally.
2. Add one or more images(*iterate_over_images()*, *add_image()*, or even patches of an image (*add_segments()*). 
   
   **Note**: If you decide to add patches, make sure that no segment in a patch is overlapping the edge of the patch. 
   Otherwise, the other parts of the segment will be considered an unique entity.
   
3. (Optional) Add any kind of metadata you want to the Labeling as additional information supplied by you
4. Get the Labeling for further use(*get_result()*) or save (*save_result()*) it to a specified path. 
   Both methods return an image and a bson container.

### Information
The project is open-source. If you find problems or have ideas for improvement, please create an issue on GitHub, or contribute yourself!