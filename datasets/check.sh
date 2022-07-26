if [ -f "$1/$1.mtx" ]; then
  mv $1/$1.mtx $1.mtx
fi
if [ -f "$1/$1.csr" ]; then
  mv $1/$1.csr $1.csr
fi
if [ -f "$1/$1_di.csr" ]; then
  mv $1/$1_di.csr $1_di.csr
fi
if [ -f "$1/$1_ud.csr" ]; then
  mv $1/$1_ud.csr $1_ud.csr
fi
