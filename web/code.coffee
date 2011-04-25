
$ ->
  # Canvas init
  window.viewer = new Viewer(document.getElementById("canvas"))
  window.viewer.setData('pldb')
  null

cmap = ([i, 0, 255-i] for i in [0...256])
MIX = .5
light_cmap = ([255*MIX + (1-MIX)*i, 0, 255*MIX + (1-MIX)*(255-i)] for i in [0...256])

MIN_GRAB_RANGE = 15

class window.Viewer
  constructor: (canvas) ->
    @running = false
    @ctx = null
    @imgData = null
    @canvas = canvas
    @lastInteraction = null
    @timeout = 3000
    @points = null
    @colors = null
    @screenX = null; @screenY = null

  init: (data) ->
    @points = data.points
    @names = data.names
    @colors = (Math.floor(255*Math.random()) for i in [0...@points.length])
    w = @canvas.width; h = @canvas.height

    @projection = new Projection(data.ndim, w, h)
    @screenX = new Uint16Array(@points.length)
    @screenY = new Uint16Array(@points.length)
    @distToMouse = new Float32Array(@points.length)
    @order = [0...@points.length]
    @labelMask = new Uint8Array(w*h)
    @heldIdx = null
    @ctx = @canvas.getContext "2d"
    @imgData = @ctx.getImageData(0, 0, w, h)

    # Set up animation loop and timers
    @lastInteraction = new Date
    @refresh()
    @mouseButton = 0
    @lastMouseX = w/2; @lastMouseY = h/2
    $(@canvas)
      .mousemove($.proxy(@onMouseMove, this))
      .mousedown($.proxy(@onMouseDown, this))
      .mouseup($.proxy(@onMouseUp, this))
      .mousewheel($.proxy(@onMouseWheel, this))
    null

  refresh: ->
    @running = true
    requestAnimFrame($.proxy(@tick, this), @canvas)

  _getBinaryAjax: (url, callback) ->
    xhr = new XMLHttpRequest()
    xhr.open('GET', url, true)
    xhr.overrideMimeType 'text/plain; charset=x-user-defined'
    xhr.onreadystatechange = ->
      if xhr.readyState == 4
        callback(parseCoords(xhr.responseText))
    xhr.send null

  setData: (url) ->
    data = {}
    maybeInit = =>
      @init data if (data.points? and data.names?)
    
    @_getBinaryAjax url+'.coords', (coords) =>
      data.ndim = coords.ndim
      data.points = coords.data
      maybeInit()

    $.ajax
      url: url+'.names'
      dataType: 'text'
      success: (names) =>
        data.names = names.split('\n')
        maybeInit()

  _interacted: ->
    @lastInteraction = new Date
    if (!@running)
      console.log('resuming')
      @refresh()
    null

  onMouseMove: (e) ->
    mX = e.pageX-@canvas.offsetLeft-1; mY = e.pageY-@canvas.offsetTop-1
    if @mouseButton == 1
      if @heldIdx?
        @projection.moveTowards @points[@heldIdx], mX, mY
    else if @mouseButton == 3
      @projection.shiftOrigin mX - @lastMouseX, mY - @lastMouseY
    @lastMouseX = mX; @lastMouseY = mY
    @_interacted()
    null

  onMouseDown: (e) ->
    mX = e.pageX-@canvas.offsetLeft-1; mY = e.pageY-@canvas.offsetTop-1
    @mouseButton = e.which
    if @mouseButton == 1
      # Grab the point closest to the mouse pointer
      @heldIdx = @order[0]
      console.log('grab', @order[0], ' = ', @names[@order[0]], @distToMouse[@order[0]])
    @lastMouseX = mX; @lastMouseY = mY
    @_interacted()
    false

  onMouseUp: (e) ->
    mX = e.pageX-@canvas.offsetLeft-1; mY = e.pageY-@canvas.offsetTop-1
    @mouseButton = 0
    @heldIdx = null
    @lastMouseX = mX; @lastMouseY = mY

  onMouseWheel: (e, delta) ->
    mX = e.pageX-@canvas.offsetLeft-1; mY = e.pageY-@canvas.offsetTop-1
    console.log('wheel', delta)
    @projection.zoomAroundBy(mX, mY, -20*delta)
    @_interacted()
    false

  tick: ->
    @projection.tick()

    # Cache really common references
    project = @projection.makeProjector()
    points = @points; colors = @colors
    screenX = @screenX; screenY = @screenY
    w = @imgData.width; h = @imgData.height; arr = @imgData.data

    times = {start: new Date}

    # Clear the canvas.
    for offset in [0...w*h*4]
      arr[offset] = 255

    times.cleared = new Date

    # Project all the points.
    for i in [0...points.length]
      vec = points[i]
      [x, y] = project vec
      screenX[i] = x; screenY[i] = y

    times.projected = new Date

    # Draw points.
    start = new Date
    for i in [0...points.length]
      x = screenX[i]; y = screenY[i]
      unless (0 < x < w and 0 < y < h)
        continue
      cidx = colors[i]
      color = cmap[cidx]
      light_color = light_cmap[cidx]
      offset = 4*(x + w*y);
      arr[offset] = color[0]; arr[offset+1] = color[1]; arr[offset+2] = color[2]; arr[offset+3] = 255;
      arr[offset-4] = light_color[0]; arr[offset-4+1] = light_color[1]; arr[offset-4+2] = light_color[2]; arr[offset-4+3] = 64;
      arr[offset+4] = light_color[0]; arr[offset+4+1] = light_color[1]; arr[offset+4+2] = light_color[2]; arr[offset+4+3] = 64;
      arr[offset-4*w] = light_color[0]; arr[offset-4*w+1] = light_color[1]; arr[offset-4*w+2] = light_color[2]; arr[offset-4*w+3] = 64;
      arr[offset+4*w] = light_color[0]; arr[offset+4*w+1] = light_color[1]; arr[offset+4*w+2] = light_color[2]; arr[offset+4*w+3] = 64;
    @ctx.putImageData(@imgData, 0, 0)

    times.drawn = new Date

    # Compute distance to mouse
    mX = @lastMouseX; mY = @lastMouseY; distToMouse = @distToMouse
    for i in [0...points.length] by 1
      x = screenX[i]; y = screenY[i]
      dx = x-mX; dy = y-mY
      distToMouse[i] = Math.sqrt(dx*dx+dy*dy)

    times.distToMouse = new Date
    
    # Sort by distance to mouse
    compare = (a, b) ->
      distToMouse[a] - distToMouse[b]
    @order.sort(compare)

    times.sort = new Date

    names = @names
    #@ctx.shadowOffsetX=2; @ctx.shadowOffsetY = 2;
    #@ctx.shadowBlur = 1; @ctx.shadowColor = "rgba(0,0,0,0.5)";
    for offset in [0...w*h] by 1
      @labelMask[offset] = 0
    nLabels = 0
    for i in [0...points.length]
      idx = @order[i]
      x = screenX[idx]; y = screenY[idx]
      unless (0 < x < w and 0 < y < h)
        continue
      if @labelMask[y*w+x]
        continue
      textWidth = @ctx.measureText(names[idx]).width
      #console.log names[idx], textWidth
      left = Math.floor(x-textWidth/2)
      right = left+textWidth
      if @labelMask[(y-6)*w+x] or @labelMask[y*w+left] or @labelMask[y*w+right] or @labelMask[(y-12)*w+left] or @labelMask[(y-12)*w+right]
        continue
      @ctx.fillStyle = (if idx == @heldIdx then '#ff0000' else '#000000')
      @ctx.fillText(names[idx], left, y)
      nLabels++
      for lmY in [y...y-12]
        offset = lmY * w
        for lmX in [left...right] by 1
          if 0 < lmX < w
            @labelMask[offset + lmX] = 1

    # Visualize the mask
    if false
      imgd = @ctx.getImageData(0, 0, w, h)
      for offset in [0...w*h]
        if @labelMask[offset]
          imgd.data[4*offset] += 64
          imgd.data[4*offset+1] -= 64
          imgd.data[4*offset+2] -= 64
          imgd.data[4*offset+3] = 255
      @ctx.putImageData(imgd, 0, 0)

    times.labeling = new Date

    if new Date - @lastInteraction > @timeout
      console.log('stopping')
      @running = false
    else
      @refresh()
    null


class Projection
  constructor: (k, width, height) ->
    @k = k; @width = width; @height = height
    @x_vec = new Float32Array(@k)
    @y_vec = new Float32Array(@k)
    @x_tgt = new Float32Array(@k)
    @y_tgt = new Float32Array(@k)
    @reset()

  reset: ->
    for i in [0...@k]
      @x_vec[i] = @x_tgt[i] = 0
      @y_vec[i] = @y_tgt[i] = 0
    @x_tgt[0] = 1.0
    @y_tgt[1] = 1.0
    @origin_x = @width/2
    @origin_y = @height/2
    @zoom = 100

  tick: ->
    for i in [0...@k]
      @x_vec[i] = @x_vec[i] * .5 + @x_tgt[i] * .5
      @y_vec[i] = @y_vec[i] * .5 + @y_tgt[i] * .5

  shiftOrigin: (deltaX, deltaY) ->
    @origin_x += deltaX
    @origin_y += deltaY

  zoomAroundBy: (x, y, deltaZoom) ->
    # in projection coordinates...
    projX = (x-@origin_x)/@zoom
    projY = (y-@origin_y)/@zoom

    @zoom *= Math.pow(.995, deltaZoom)

    @origin_x = -(projX * @zoom - x)
    @origin_y = -(projY * @zoom - y)


  # Move a k-space vector towards a screen-point target
  moveTowards: (vec, targetX, targetY) ->
    # Convert to projection coordinates
    targetX = (targetX - @origin_x)/@zoom
    targetY = (targetY - @origin_y)/@zoom

    # Find out where the point originally should have projected to.
    origX = dot(@x_tgt, vec); origY = dot(@y_tgt, vec)

    # delta = (target - orig) / dot(vec, vec)
    vecMag = dot(vec, vec)
    deltaX = (targetX - origX) / vecMag
    deltaY = (targetY - origY) / vecMag

    # target_matrix += delta[np.newaxis,:] * vec[:,np.newaxis]
    for i in [0...@k]
      @x_tgt[i] += deltaX * vec[i]
      @y_tgt[i] += deltaY * vec[i]

    magnitude = deltaX*deltaX + deltaY*deltaY
    power = Math.atan(magnitude)/10
    @orthogonalize power

  # Use normalization and Gram-Schmidt orthogonalization to ensure that
  # this projection represents a valid rotation of the component space,
  # not a dilation or skew.
  #
  # Actually, now it only does that when power=1. When power is less, the
  # space will not enforce strict rotations, it will only make the space
  # wobble back toward orthogonality.
  orthogonalize: (power) ->
    targetX = @x_tgt; targetY = @y_tgt
    prevX = copyVec(targetX); prevY = copyVec(targetY)
    normalizeInPlace(targetX)
    commonMag = dot(targetX, targetY) # NOTE: in the Python code the magnitude was y_vec, not y_tgt
    for i in [0...@k]
      targetY[i] -= targetX[i] * commonMag
    normalizeInPlace(targetY)
    # target = target*(power) + prev*(1-power)
    for i in [0...@k]
      targetX[i] = targetX[i]*power + prevX[i]*(1-power)
      targetY[i] = targetY[i]*power + prevY[i]*(1-power)

  makeProjector: ->
    x_vec = @x_vec; y_vec = @y_vec
    origin_x = @origin_x; origin_y = @origin_y
    zoom = @zoom
    return (vec) ->
      x = dot(x_vec, vec); y = dot(y_vec, vec)
      [x*zoom+origin_x, y*zoom+origin_y]

  project: (vec) ->
    (@makeProjector()) vec


parseCoords = (data) ->
  ndim = readUInt32(data, 0)
  nitems = readUInt32(data, 4)
  offset = 8
  arr = []
  for item in [0...nitems] by 1
    entry = new Float32Array(ndim)
    arr.push(entry)
    for dim in [0...ndim] by 1
      entry[dim] = readFloat32(data, offset)
      offset += 4
  return {ndim: ndim, nitems: nitems, data: arr}

dot = (vec1, vec2) ->
  s = 0
  for i in [0...vec1.length] by 1
    s += vec1[i] * vec2[i]
  s

normalizeInPlace = (vec) ->
  mag = Math.sqrt dot(vec, vec)
  for i in [0...vec.length]
    vec[i] /= mag
  null

copyVec = (vec) ->
  v2 = new Float32Array(vec.length)
  for i in [0...vec.length]
    v2[i] = vec[i]
  v2
