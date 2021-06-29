import argparse
import gzip
import os
import sys
import urllib
import zipfile

try:
    from urllib.error import URLError
    from urllib.request import urlretrieve
except ImportError:
    from urllib2 import URLError
    from urllib import urlretrieve

RESOURCES = [
    "train.zip",
    "valid.zip",
    "test.zip",
]

def report_download_progress(chunk_number, chunk_size, file_size):
    if file_size != -1:
        percent = min(1, (chunk_number * chunk_size) / file_size)
        bar = "#" * int(64 * percent)
        sys.stdout.write("\r0% |{:<64}| {}%".format(bar, int(percent * 100)))


def download(destination_path, url, quiet):
    if os.path.exists(destination_path):
        if not quiet:
            print("{} already exists, skipping ...".format(destination_path))
    else:
        print("Downloading {} ...".format(url))
        try:
            hook = None if quiet else report_download_progress
            urlretrieve(url, destination_path, reporthook=hook)
        except URLError:
            raise RuntimeError("Error downloading resource!")
        finally:
            if not quiet:
                # Just a newline.
                print()


def unzip(zipped_path, quiet):
    unzipped_path = os.path.splitext(zipped_path)[0]
    print(unzipped_path)
    if os.path.exists(unzipped_path):
        if not quiet:
            print("{} already exists, skipping ... ".format(unzipped_path))
        return
    with zipfile.ZipFile(f"{unzipped_path}.zip", 'r') as zip_ref:
        zip_ref.extractall(unzipped_path)
        if not quiet:
            print("Unzipped {} ...".format(zipped_path))
def main():
    parser = argparse.ArgumentParser(
        description="Download the MNIST dataset from the internet"
    )
    parser.add_argument(
        "-d", "--destination", default=".", help="Destination directory"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Don't report about progress"
    )
    options = parser.parse_args()

    if not os.path.exists(options.destination): os.makedirs(options.destination)
    try:
        for resource in RESOURCES:
            path = os.path.join(options.destination, resource)
            url = (
                    f"https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/{resource}"
                )
            download(path, url, True)
            unzip(path, False)
    except KeyboardInterrupt:
        print("Interrupted")

if __name__ == "__main__":
    main()
