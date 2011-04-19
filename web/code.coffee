$ ->
  canvas = document.getElementById "canvas"
  console.log(canvas.width, canvas.height)
  setTimeout tick, 100

points = ([900*Math.random(), 900*Math.random(), Math.floor(255*Math.random())] for i in [0...10000])
cmap = ([i, 0, 255-i] for i in [0...256])
MIX = .5
light_cmap = ([255*MIX + (1-MIX)*i, 0, 255*MIX + (1-MIX)*(255-i)] for i in [0...256])

tick = ->
  start = new Date
  canvas = document.getElementById("canvas")
  w=canvas.width; h=canvas.height
  ctx = canvas.getContext "2d"
  ctx.clearRect(0, 0, w, h);
  # ctx.fillRect(0, 0, w, h);
  imgData = ctx.getImageData(0, 0, w, h);
  console.log(imgData.data.length);
  arr = imgData.data
  for pt in points
    [x, y, cidx] = pt
    color = cmap[cidx]
    light_color = light_cmap[cidx]
    offset = Math.floor(4*(x + w*(Math.floor(y))));
    arr[offset] = color[0]; arr[offset+1] = color[1]; arr[offset+2] = color[2]; arr[offset+3] = 255;
    arr[offset-4] = light_color[0]; arr[offset-4+1] = light_color[1]; arr[offset-4+2] = light_color[2]; arr[offset-4+3] = 64;
    arr[offset+4] = light_color[0]; arr[offset+4+1] = light_color[1]; arr[offset+4+2] = light_color[2]; arr[offset+4+3] = 64;
    arr[offset-4*w] = light_color[0]; arr[offset-4*w+1] = light_color[1]; arr[offset-4*w+2] = light_color[2]; arr[offset-4*w+3] = 64;
    arr[offset+4*w] = light_color[0]; arr[offset+4*w+1] = light_color[1]; arr[offset+4*w+2] = light_color[2]; arr[offset+4*w+3] = 64;
  ctx.putImageData(imgData, 0, 0)
  console.log(new Date - start)

  start = new Date
  for i in [0...points.length/10]
    pt = points[i]
    ctx.fillText("pt", Math.floor(pt[0]), Math.floor(pt[1]));
  console.log(new Date - start);
  null
