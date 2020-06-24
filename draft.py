{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('..')\n",
    "\n",
    "import numpy as np\n",
    "import torch\n",
    "\n",
    "import lib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'os' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-2-0e0ad79490af>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0mDATA_DIR\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m'./data/SIFT100K'\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 4\u001b[0;31m \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mos\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpath\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mexists\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mDATA_DIR\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0;32mFalse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      5\u001b[0m     \u001b[0;32massert\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mDATA_DIR\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mendswith\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mos\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msep\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'please do not put \"/\" at the end of DATA_DIR'\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m     \u001b[0mget_ipython\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msystem\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'mkdir -p {DATA_DIR}'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'os' is not defined"
     ]
    }
   ],
   "source": [
    "#DATA TODO\n",
    "DATA_DIR = './data/SIFT100K'\n",
    "\n",
    "if not os.path.exists(DATA_DIR) and False:\n",
    "    assert not DATA_DIR.endswith(os.sep), 'please do not put \"/\" at the end of DATA_DIR'\n",
    "    !mkdir -p {DATA_DIR}\n",
    "    !curl -L https://www.dropbox.com/sh/gdb8aedbv4qt2b2/AAAwo1GCUXfI7FwUYvNFxV92a?dl=1 > {DATA_DIR}/sift_100k.zip\n",
    "    !cd {DATA_DIR} && unzip sift_100k.zip"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "a = dict()\n",
    "a['he'] = 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "a\n",
      "b\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<__main__.Test at 0x7f47588aa7d0>"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "class Test:\n",
    "    def __init__(self, mode=1, **kwargs):\n",
    "        if mode == 1:\n",
    "            for k in kwargs:\n",
    "                self.k = 1\n",
    "                print(k)\n",
    "        else:\n",
    "            print(0)\n",
    "    \n",
    "t = Test(a=3, b=10)\n",
    "t"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
