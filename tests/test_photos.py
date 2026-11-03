from PIL import Image

from profilometer_comparison.photos import PhotoStore, fit_photo


def _photo(path, size=(400, 300), colour=(200, 100, 50), orientation=None):
    img = Image.new("RGB", size, colour)
    if orientation:
        exif = img.getexif()
        exif[0x0112] = orientation
        img.save(path, exif=exif)
    else:
        img.save(path)
    return path


def test_find_photos_by_sample_name(tmp_path):
    before, after = tmp_path / "before-photos", tmp_path / "after-photos"
    before.mkdir(), after.mkdir()
    _photo(before / "M1.jpg")
    _photo(before / "M10.JPG")
    _photo(before / "m2.png")
    _photo(after / "M1.jpeg")
    (before / "notes.txt").write_text("not a photo")
    store = PhotoStore(tmp_path / "results")
    assert not store.configured
    store.set_folders(before, after)
    assert store.configured
    assert store.find("M1", "before").name == "M1.jpg"  # not M10
    assert store.find("M10", "before").name == "M10.JPG"
    assert store.find("M2", "before").name == "m2.png"  # case does not matter
    assert store.find("M1", "after").name == "M1.jpeg"
    assert store.find("M2", "after") is None
    assert store.find("M3", "before") is None
    # high-res rescans use the photo of their sample
    assert store.find("M1-high-res", "before").name == "M1.jpg"

    reopened = PhotoStore(tmp_path / "results")  # folders are remembered in the results folder
    assert reopened.find("M1", "after").name == "M1.jpeg"


def test_photo_name_with_suffix(tmp_path):
    folder = tmp_path / "p"
    folder.mkdir()
    _photo(folder / "M7-before.jpg")
    _photo(folder / "M70.jpg")
    store = PhotoStore(tmp_path / "results")
    store.set_folders(folder, None)
    assert store.find("M7", "before").name == "M7-before.jpg"
    assert store.find("M7", "after") is None  # no after folder set


def test_set_one_folder_keeps_the_other(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    store = PhotoStore(tmp_path / "results")
    store.set_folders(a, b)
    store.set_folders(None, tmp_path)  # None: keep
    assert store.folders == {"before": a, "after": tmp_path}


def test_load_applies_exif_rotation_and_size(tmp_path):
    folder = tmp_path / "p"
    folder.mkdir()
    _photo(folder / "M1.jpg", size=(800, 600), orientation=6)  # camera held upright
    store = PhotoStore(tmp_path / "results")
    store.set_folders(folder, None)
    img = store.load("M1", "before", 200)
    assert img.height > img.width  # turned upright
    assert max(img.size) <= 200
    assert store.load("M5", "before", 200) is None


def test_fit_photo_keeps_aspect():
    img = Image.new("RGB", (4000, 3000))
    assert fit_photo(img, 400, 400).size == (400, 300)
    assert fit_photo(img, 400, 150).size == (200, 150)


def test_rotate_photo_per_sample_and_side(tmp_path):
    folder = tmp_path / "p"
    folder.mkdir()
    img = Image.new("RGB", (400, 300), (0, 0, 0))
    img.putpixel((0, 0), (255, 0, 0))  # marker in the top-left corner
    img.save(folder / "M1.png")
    _photo(folder / "M2.png")
    store = PhotoStore(tmp_path / "results")
    store.set_folders(folder, folder)
    assert store.rotation("M1", "before") == 0
    assert store.rotate("M1", "before") == 90
    turned = store.load("M1", "before", 1000)
    assert turned.size == (300, 400)  # 90° clockwise: the top-left corner moves to the top-right
    assert turned.getpixel((299, 0)) == (255, 0, 0)
    assert store.load("M1", "after", 1000).size == (400, 300)  # the after photo is not turned
    assert store.load("M2", "before", 1000).size == (400, 300)  # nor other samples
    for _ in range(3):
        store.rotate("M1", "before")
    assert store.rotation("M1", "before") == 0  # four turns: back to the start
    store.rotate("M1", "after")
    reopened = PhotoStore(tmp_path / "results")  # remembered in photos.json
    assert reopened.rotation("M1", "after") == 90
    assert reopened.folders["before"] == folder
    reopened.set_folders(None, folder)  # choosing folders keeps the rotations
    assert PhotoStore(tmp_path / "results").rotation("M1", "after") == 90
