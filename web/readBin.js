// From http://fhtr.blogspot.com/2009/12/3d-models-and-parsing-binary-data-with.html

// read big-endian (network byte order) unsigned 32-bit int from data, at offset
readUInt32 = function(data, offset) {
  return ((data.charCodeAt(offset) & 0xFF) << 24) +
         ((data.charCodeAt(offset+1) & 0xFF) << 16) +
         ((data.charCodeAt(offset+2) & 0xFF) << 8) +
         (data.charCodeAt(offset+3) & 0xFF);
}

// read big-endian (network byte order) 32-bit float
readFloat32 = function(data, offset) {
  var b1 = data.charCodeAt(offset) & 0xFF,
      b2 = data.charCodeAt(offset+1) & 0xFF,
      b3 = data.charCodeAt(offset+2) & 0xFF,
      b4 = data.charCodeAt(offset+3) & 0xFF;
  var sign = 1 - (2*(b1 >> 7));                     // sign = bit 0
  var exp = (((b1 << 1) & 0xff) | (b2 >> 7)) - 127; // exponent = bits 1..8
  var sig = ((b2 & 0x7f) << 16) | (b3 << 8) | b4;   // significand = bits 9..31
  if (sig == 0 && exp == -127)
    return 0.0;
  return sign * (1 + sig * Math.pow(2, -23)) * Math.pow(2, exp);
}
