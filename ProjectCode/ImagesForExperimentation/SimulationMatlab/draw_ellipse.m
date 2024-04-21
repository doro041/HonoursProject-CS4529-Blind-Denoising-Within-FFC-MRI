function draw_ellipse(a,b,posx,posy,level,alpha)

ha = gca;
x=linspace(-a,a,160); y = b*sqrt(1-(x./a).^2);
x = [x fliplr(x)]; y = [-y y]; z = ones(size(x))*level;
rot = [cosd(alpha),-sind(alpha);sind(alpha),cosd(alpha)];
for i = 1:numel(x)
    v = rot*[x(i);y(i)];
    x(i) = v(1);
    y(i) = v(2);
end

patch(x+posx,y+posy,z,'EdgeColor','none')

