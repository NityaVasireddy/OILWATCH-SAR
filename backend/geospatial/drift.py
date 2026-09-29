import math

def backtrack(lat,lon,wind_speed,wind_dir,current_speed,current_dir,duration_hours,steps=24):
 if any(v is None for v in [lat,lon,wind_speed,wind_dir,current_speed,current_dir,duration_hours]) or duration_hours<=0: raise ValueError('Insufficient drift inputs.')
 pts=[(lat,lon)]; dt=duration_hours/steps
 # direction is degrees clockwise from north; backtracking subtracts the forward displacement.
 for _ in range(steps):
  def comp(speed,deg):
   r=math.radians(deg);return speed*math.sin(r),speed*math.cos(r)
  ex,ny=comp(wind_speed,wind_dir);cx,cy=comp(current_speed,current_dir); east=(ex+cx)*dt; north=(ny+cy)*dt
  lat-=north/111.32;lon-=east/(111.32*max(math.cos(math.radians(lat)),1e-6));pts.append((lat,lon))
 return {'method':'Simplified drift model','estimated_release_region':{'latitude':pts[-1][0],'longitude':pts[-1][1]},'time_window_hours':duration_hours,'drift_path':pts,'assumptions':['Constant wind and current vectors over the backtracking window.','Spherical Earth approximation.','No diffusion, coastline interaction, weathering, or wave-induced Stokes drift.']}
