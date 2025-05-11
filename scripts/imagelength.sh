for object_name in $(ls data/Diva360); do
    # Get the length of the image
    length=$(ls data/Diva360/$object_name/cam00 | wc -l)
    echo "Length of images in $object_name: $length"
done