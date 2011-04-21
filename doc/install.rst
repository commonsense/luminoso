Installation
============

Luminoso 2 is currently provided as source code only. On the plus side, this
means it can be installed on any system that meets the requirements. On the
minus side, this means it may take a considerable amount of effort to install
on a system that is not set up for Python and C development.

Luminoso 2 is easiest to install on Linux, reasonable to install on Mac, and
unreasonable to install on Windows.

A complete installation guide for Ubuntu
----------------------------------------
We recommend running Luminoso on Ubuntu Linux. If you are on Ubuntu, the
following two commands are all you need to install Luminoso with all its
features::

    sudo aptitude install python-dev python-setuptools python-pip build-essential python-numpy mecab mecab-ipadic-utf8
    sudo pip install csc-pysparse luminoso2

That's all! To upgrade Luminoso later, run::

    sudo pip install --upgrade luminoso2

Development version
```````````````````

If you want to use the Git repository so that you can easily get the latest
updates or edit the Luminoso code, do this instead::

    sudo aptitude install python-dev python-setuptools python-pip build-essential python-numpy python-numpmecab mecab-ipadic-utf8 git
    git clone git://github.com/commonsense/luminoso.git -b newstudy luminoso2
    cd luminoso2
    sudo python setup.py develop

To update it later, cd to the `luminoso2` directory, and run `git pull`. Major
updates may require you to run `sudo python setup.py develop` again.

Some Python programmers may want to install Luminoso in a `virtualenv`.  In
this case, run `python setup.py develop`, without the `sudo`.

Installation on a Mac
---------------------

The things you need to make sure are set up first are Python, NumPy, Xcode, and
Pip.

**Python**: On Mac OS 10.6 or later, the version of Python that comes with the
OS can run Luminoso. This should make things quite straightforward. The issue
we encounter most often is that you're lacking a Python interpreter, but that
you have *too many* Python interpreters.

If you have installed a version of Python separate from the one that comes with
the OS -- either by downloading it from python.org or by using MacPorts -- you
now have two programs that answer to "python". This can work just fine, but
it's up to you to keep track of which one you're using and make sure you're
consistent.

Usually you're in this position because you know exactly how you want your
Python environment to be set up. If you have two Pythons because someone else's
install instructions told you to install another one, and now you don't know
which scripts work with which, we deeply apologize.

**Xcode**: You will need Xcode 3 or later, so that you can compile C code. It's
on your Mac OS installation DVD, but you may not have installed it. Sorry, it's
important.

When choosing which Xcode components to install, make sure to install "C++
development" and the "10.4u SDK".

If you can't find your installation DVD, you may be able to download Xcode from
the Web, or Apple sells a newer version for $5 in the App Store. Of course,
it's better to find the DVD version because it's free and it's not a
multi-gigabyte download.

**pip**: Pip is our system of choice for installing Python packages and their
dependencies. It tends to handle our code slightly better than `easy_install`.
Your system Python comes with `easy_install`, so one way to get Pip (*for that
version of Python only*) is to type at the command prompt::

    sudo easy_install pip

**NumPy**: NumPy, for Numeric Python, is a very important library for
performing fast numerical operations in Python. You can download an installer
for your version of Python from http://numpy.scipy.org.

**MeCab** (optional): If you want to handle Japanese text, you should download
and install MeCab from http://sourceforge.net/projects/mecab/, and then install
its dictionary from http://sourceforge.net/projects/mecab/files/mecab-ipadic/.
You no longer need the `mecab-python` interface; we work with `mecab` directly.

Finally::

    sudo pip install csc-pysparse luminoso2

Installation on Windows
-----------------------
Luminoso2 is cross-platform Python code that works on Windows. Its dependencies
may be more difficult.

Unfortunately, Windows systems are all different from each other, *especially*
ones that are set up for development, so the best we can give you for now is a
vague outline of what to do. (We plan to eventually package up some
pre-compiled code for Windows.)

You will need a Python setup that works at your favorite Windows command line
(this is non-trivial), a C compiler such as Visual Studio or `mingw32` that
Python knows how to use, and Pip and NumPy installed for your version of
Python.

If you have a 64-bit version of Windows, we recommend installing 32-bit Python
anyway. If you know how to compile 64-bit Python extensions on Windows without
expensive development tools, please let us know.

Once you have those, use your command line to run `pip install csc-pysparse
luminoso2`. If it spews error messages, you're missing one of the things above.

