function previewImage(event) {
    const file = event.target.files[0];
    const preview = document.getElementById("preview");

    if (file) {
        preview.src = URL.createObjectURL(file);
        preview.classList.remove("d-none");
    }
}
