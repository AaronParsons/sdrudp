# Installation
To set up the SDR, follow the instructions on [AstroBaki](https://casper.astro.berkeley.edu/astrobaki/index.php/Setting_Up_Your_Raspberry_Pi). Specfically, install the dependencies like this:

    git clone git@github.com:AaronParsons/librtlsdr.git
    cd librtlsdr
    autoreconf -i
    ./configure
    make
    sudo make install
    sudo ldconfig
    cd ..
    git clone git@github.com:AaronParsons/pyrtlsdr.git
    cd pyrtlsdr
    pip install .

Then clone and install this repository:

    git clone git@github.com:AaronParsons/sdrudp.git
    cd sdrudp
    pip install .
  
 This automatically installs the dependencies ``numpy``, ``matplotlib``, and ``lz4``.
